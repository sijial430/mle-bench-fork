import os
import re
import sys
import gc
import math
import random
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from tqdm import tqdm

# Configuration
INPUT_DIR = "./input"
TRAIN_FILE = os.path.join(INPUT_DIR, "train.csv")
TEST_FILE = os.path.join(INPUT_DIR, "test.csv")
SAMPLE_SUB_FILE = os.path.join(INPUT_DIR, "sample_submission.csv")
SUBMISSION_DIR = "./submission"
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

TOP_K_TAGS = 300  # number of most frequent tags to model
MAX_SAMPLES = 120000  # maximum number of training examples to use
RANDOM_STATE = 42
N_FOLDS = 5
TOP_PRED = 3  # number of tags to output per question

os.makedirs(SUBMISSION_DIR, exist_ok=True)
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def parse_tags(tagstr):
    if pd.isna(tagstr):
        return []
    s = tagstr.strip()
    if not s:
        return []
    # Two common formats: "<tag1><tag2>" or "tag1 tag2"
    if s.startswith("<") and ">" in s:
        # split by '><' after stripping leading/trailing angle brackets
        s2 = s.strip()
        s2 = s2.strip("<>")
        return [t for t in s2.split("><") if t]
    else:
        # assume whitespace-separated
        return [t for t in s.split() if t]


print("Step 1: Scan train file to count tag frequencies...")
tag_counter = Counter()
# read in chunks to avoid loading whole file
chunksize = 100000
usecols = ["Tags"]
for chunk in pd.read_csv(TRAIN_FILE, usecols=usecols, chunksize=chunksize):
    for tags in chunk["Tags"].fillna("").astype(str):
        for t in parse_tags(tags):
            tag_counter[t] += 1
print(f"Found {len(tag_counter)} unique tags in training data.")

top_tags = [t for t, _ in tag_counter.most_common(TOP_K_TAGS)]
top_tag_set = set(top_tags)
print(f"Selected top {len(top_tags)} tags (most common).")

print("Step 2: Collect up to MAX_SAMPLES examples that include at least one top tag...")
collected = []
collected_tags = []
collected_ids = []
collected_texts = []

# We'll read again and keep rows that have at least one top tag
cols = ["Id", "Title", "Body", "Tags"]
for chunk in pd.read_csv(TRAIN_FILE, usecols=cols, chunksize=chunksize):
    for idx, row in chunk.iterrows():
        tags = parse_tags(row["Tags"] if not pd.isna(row["Tags"]) else "")
        tags_in_top = [t for t in tags if t in top_tag_set]
        if tags_in_top:
            # combine title and body
            title = row["Title"] if not pd.isna(row["Title"]) else ""
            body = row["Body"] if not pd.isna(row["Body"]) else ""
            text = (str(title) + " " + str(body)).strip()
            collected_ids.append(row["Id"])
            collected_texts.append(text)
            collected_tags.append(tags_in_top)
            if len(collected_ids) >= MAX_SAMPLES:
                break
    if len(collected_ids) >= MAX_SAMPLES:
        break

n_collected = len(collected_ids)
print(f"Collected {n_collected} training examples with top tags (limit {MAX_SAMPLES}).")
if n_collected == 0:
    raise RuntimeError("No training examples collected; check parsing.")

# Shuffle collected examples
perm = np.arange(n_collected)
np.random.shuffle(perm)
collected_ids = [collected_ids[i] for i in perm]
collected_texts = [collected_texts[i] for i in perm]
collected_tags = [collected_tags[i] for i in perm]

# Prepare multilabel binarizer for only top tags
mlb = MultiLabelBinarizer(classes=top_tags)
Y = mlb.fit_transform(collected_tags)
print("Binarized tags shape:", Y.shape)

print(
    "Step 3: Vectorize text with TfidfVectorizer (unigrams+bigrams, capped features)..."
)
tfidf = TfidfVectorizer(
    max_features=100000, ngram_range=(1, 2), stop_words="english", min_df=3
)
X = tfidf.fit_transform(collected_texts)
print("TF-IDF matrix shape:", X.shape)

# free memory
del collected_texts
gc.collect()

print(
    f"Step 4: {N_FOLDS}-fold cross-validation with One-vs-Rest MultinomialNB and sample-wise F1"
)
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
f1s = []
fold = 0
for train_idx, val_idx in kf.split(X):
    fold += 1
    print(f" Training fold {fold}...")
    X_train = X[train_idx]
    X_val = X[val_idx]
    Y_train = Y[train_idx]
    Y_val = Y[val_idx]

    clf = OneVsRestClassifier(MultinomialNB(alpha=0.1), n_jobs=-1)
    clf.fit(X_train, Y_train)
    # predict probabilities and take top-PRED per sample
    try:
        probs = clf.predict_proba(X_val)
    except Exception:
        # fallback: use decision_function then sigmoid
        dec = clf.decision_function(X_val)
        probs = 1.0 / (1.0 + np.exp(-dec))

    # For each sample, pick top TOP_PRED tags
    top_preds = np.zeros_like(probs, dtype=int)
    topk_idx = np.argsort(-probs, axis=1)[:, :TOP_PRED]
    rows = np.arange(probs.shape[0])[:, None]
    top_preds[rows, topk_idx] = 1

    f1 = f1_score(Y_val, top_preds, average="samples")
    print(f"  Fold {fold} sample-wise F1 = {f1:.6f}")
    f1s.append(f1)

mean_f1 = float(np.mean(f1s))
print(f"\nMean sample-wise F1 across {N_FOLDS} folds = {mean_f1:.6f}")

print(
    "Step 5: Retrain on all collected training data and predict on test set in batches..."
)
clf_full = OneVsRestClassifier(MultinomialNB(alpha=0.1), n_jobs=-1)
clf_full.fit(X, Y)
del X, Y
gc.collect()

# Prepare to read test and predict in chunks
test_cols = ["Id", "Title", "Body"]
test_reader = pd.read_csv(TEST_FILE, usecols=test_cols, chunksize=5000)

out_ids = []
out_tags = []

for chunk in tqdm(test_reader, desc="Predicting test chunks"):
    ids = chunk["Id"].tolist()
    texts = (
        (
            chunk["Title"].fillna("").astype(str)
            + " "
            + chunk["Body"].fillna("").astype(str)
        )
    ).tolist()
    X_test = tfidf.transform(texts)
    try:
        probs = clf_full.predict_proba(X_test)
    except Exception:
        dec = clf_full.decision_function(X_test)
        probs = 1.0 / (1.0 + np.exp(-dec))
    # choose top TOP_PRED tags per row
    topk_idx = np.argsort(-probs, axis=1)[:, :TOP_PRED]
    for i, idxs in enumerate(topk_idx):
        tags = [mlb.classes_[j] for j in idxs]
        out_ids.append(ids[i])
        out_tags.append(" ".join(tags))

# Write submission file
sub_df = pd.DataFrame({"Id": out_ids, "Tags": out_tags})
sub_df.to_csv(SUBMISSION_FILE, index=False)
print(f"Submission saved to: {SUBMISSION_FILE}")

# Print final metric
print(f"\nFINAL METRIC (mean sample-wise F1 from CV): {mean_f1:.6f}")
