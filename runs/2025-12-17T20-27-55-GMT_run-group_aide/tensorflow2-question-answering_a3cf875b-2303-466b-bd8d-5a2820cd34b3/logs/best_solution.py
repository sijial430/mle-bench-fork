import os
import json
import random
import gc
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

# Paths
INPUT_DIR = "./input"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

train_path = os.path.join(INPUT_DIR, "simplified-nq-train.jsonl")
# Note: benchmark uses simplified-nq-kaggle-test.jsonl name in description, but code previously used this:
test_path = os.path.join(INPUT_DIR, "simplified-nq-test.jsonl")
if not os.path.exists(test_path):
    # Fallback to kaggle-style name if needed
    alt = os.path.join(INPUT_DIR, "simplified-nq-kaggle-test.jsonl")
    if os.path.exists(alt):
        test_path = alt
sample_sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")

# Parameters
MAX_TRAIN_EXAMPLES = 15000
POS_MULTIPLIER = 3
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def extract_labels_from_annotation(annotations):
    long_has_answer = 0
    short_has_answer = 0
    for ann in annotations:
        la = ann.get("long_answer", {})
        if la and la.get("start_token", -1) != -1 and la.get("end_token", -1) != -1:
            long_has_answer = 1
        sa = ann.get("short_answers", [])
        yes_no = ann.get("yes_no_answer", "NONE")
        if sa and len(sa) > 0:
            short_has_answer = 1
        if yes_no in ["YES", "NO"]:
            short_has_answer = 1
    return long_has_answer, short_has_answer


def build_sampled_dataset(
    train_path, max_examples=MAX_TRAIN_EXAMPLES, seed=RANDOM_SEED
):
    random.seed(seed)
    texts = []
    y_long = []
    y_short = []

    pos_long = neg_long = 0
    pos_short = neg_short = 0

    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            if len(texts) >= max_examples:
                break
            ex = json.loads(line)
            annotations = ex.get("annotations", [])
            la, sa = extract_labels_from_annotation(annotations)

            if la == 1:
                pos_long += 1
            else:
                neg_long += 1
            if sa == 1:
                pos_short += 1
            else:
                neg_short += 1

            allow = True
            if la == 0 and pos_long > 0 and neg_long > POS_MULTIPLIER * pos_long:
                allow = False
            if sa == 0 and pos_short > 0 and neg_short > POS_MULTIPLIER * pos_short:
                allow = False
            if not allow:
                continue

            q = ex.get("question_text", "")
            doc = ex.get("document_text", "")
            tokens = doc.split()
            doc_trunc = " ".join(tokens[:400])
            text = q + " [SEP] " + doc_trunc

            texts.append(text)
            y_long.append(la)
            y_short.append(sa)

    return texts, np.array(y_long), np.array(y_short)


def safe_f1(y_true, y_pred):
    if sum(y_true) == 0 and sum(y_pred) == 0:
        return 0.0
    return f1_score(y_true, y_pred, average="binary")


def find_best_threshold(y_true, y_prob, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 17)
    best_t = 0.5
    best_f1 = -1.0
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = safe_f1(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1


print("Building sampled dataset for training/validation...")
texts, y_long, y_short = build_sampled_dataset(train_path)
print(f"Collected {len(texts)} examples.")
print(f"Long label distribution: {Counter(y_long)}")
print(f"Short label distribution: {Counter(y_short)}")

print("Starting 5-fold cross-validation with threshold optimization...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

fold_f1_long = []
fold_f1_short = []
fold_f1_micro = []
best_thresholds_long = []
best_thresholds_short = []

strat_labels = y_long.copy()

for fold, (train_idx, val_idx) in enumerate(skf.split(texts, strat_labels), 1):
    print(f"\nFold {fold}")
    X_train_texts = [texts[i] for i in train_idx]
    X_val_texts = [texts[i] for i in val_idx]
    y_long_train, y_long_val = y_long[train_idx], y_long[val_idx]
    y_short_train, y_short_val = y_short[train_idx], y_short[val_idx]

    vectorizer_cv = TfidfVectorizer(max_features=40000, ngram_range=(1, 2), min_df=2)
    X_train = vectorizer_cv.fit_transform(X_train_texts)
    X_val = vectorizer_cv.transform(X_val_texts)

    clf_long_cv = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=200,
        random_state=RANDOM_SEED,
    )
    clf_short_cv = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=200,
        random_state=RANDOM_SEED,
    )

    clf_long_cv.fit(X_train, y_long_train)
    clf_short_cv.fit(X_train, y_short_train)

    y_long_val_prob = clf_long_cv.predict_proba(X_val)[:, 1]
    y_short_val_prob = clf_short_cv.predict_proba(X_val)[:, 1]

    # Optimize threshold per task on this fold
    t_long, _ = find_best_threshold(y_long_val, y_long_val_prob)
    t_short, _ = find_best_threshold(y_short_val, y_short_val_prob)
    best_thresholds_long.append(t_long)
    best_thresholds_short.append(t_short)

    y_long_val_pred = (y_long_val_prob >= t_long).astype(int)
    y_short_val_pred = (y_short_val_prob >= t_short).astype(int)

    f1_long = safe_f1(list(y_long_val), list(y_long_val_pred))
    f1_short = safe_f1(list(y_short_val), list(y_short_val_pred))
    y_true_all = list(y_long_val) + list(y_short_val)
    y_pred_all = list(y_long_val_pred) + list(y_short_val_pred)
    f1_micro = safe_f1(y_true_all, y_pred_all)

    fold_f1_long.append(f1_long)
    fold_f1_short.append(f1_short)
    fold_f1_micro.append(f1_micro)

    print(f"Fold {fold} F1 long (thr={t_long:.3f}): {f1_long:.6f}")
    print(f"Fold {fold} F1 short (thr={t_short:.3f}): {f1_short:.6f}")
    print(f"Fold {fold} micro F1: {f1_micro:.6f}")

avg_t_long = float(np.mean(best_thresholds_long)) if best_thresholds_long else 0.5
avg_t_short = float(np.mean(best_thresholds_short)) if best_thresholds_short else 0.5

print("\n==== Cross-Validation Results with optimized thresholds ====")
print(f"Mean F1 long:  {np.mean(fold_f1_long):.6f} (+/- {np.std(fold_f1_long):.6f})")
print(f"Mean F1 short: {np.mean(fold_f1_short):.6f} (+/- {np.std(fold_f1_short):.6f})")
print(f"Mean micro F1: {np.mean(fold_f1_micro):.6f} (+/- {np.std(fold_f1_micro):.6f})")
print(f"Average optimal threshold long:  {avg_t_long:.4f}")
print(f"Average optimal threshold short: {avg_t_short:.4f}")

# Retrain final models on full sampled dataset
print("\nTraining final models on full sampled dataset...")

print("Fitting TF-IDF vectorizer on all data...")
vectorizer = TfidfVectorizer(max_features=40000, ngram_range=(1, 2), min_df=2)
X_all = vectorizer.fit_transform(texts)

print("Training Logistic Regression for long_has_answer (full data)...")
clf_long = LogisticRegression(
    solver="liblinear", class_weight="balanced", max_iter=200, random_state=RANDOM_SEED
)
clf_long.fit(X_all, y_long)

print("Training Logistic Regression for short_has_answer (full data)...")
clf_short = LogisticRegression(
    solver="liblinear", class_weight="balanced", max_iter=200, random_state=RANDOM_SEED
)
clf_short.fit(X_all, y_short)

# Compute training-set F1 using optimized thresholds
y_long_all_prob = clf_long.predict_proba(X_all)[:, 1]
y_short_all_prob = clf_short.predict_proba(X_all)[:, 1]

y_long_all_pred = (y_long_all_prob >= avg_t_long).astype(int)
y_short_all_pred = (y_short_all_prob >= avg_t_short).astype(int)

f1_long_full = safe_f1(list(y_long), list(y_long_all_pred))
f1_short_full = safe_f1(list(y_short), list(y_short_all_pred))
y_true_all_full = list(y_long) + list(y_short)
y_pred_all_full = list(y_long_all_pred) + list(y_short_all_pred)
f1_micro_full = safe_f1(y_true_all_full, y_pred_all_full)
print("\nTraining-set F1 with optimized thresholds (overfitted estimate):")
print(f"Train F1 long:  {f1_long_full:.6f}")
print(f"Train F1 short: {f1_short_full:.6f}")
print(f"Train micro F1: {f1_micro_full:.6f}")

# Use cross-validated average thresholds as final thresholds for prediction
thr_long_final = avg_t_long
thr_short_final = avg_t_short

print(
    f"\nFinal thresholds used for prediction - long: {thr_long_final:.4f}, short: {thr_short_final:.4f}"
)

# Create submission
print("\nCreating submission predictions...")

sample_sub = pd.read_csv(sample_sub_path)
unique_ids = sample_sub["example_id"].apply(lambda x: x.split("_")[0]).unique()
id_to_rows = {}
for idx, row in sample_sub.iterrows():
    ex_id, ans_type = row["example_id"].split("_")
    id_to_rows.setdefault(ex_id, {})[ans_type] = idx

print("Loading and predicting on test data...")
test_texts = []
test_meta = []  # (example_id, long_answer_candidates, doc_len)

with open(test_path, "r", encoding="utf-8") as f:
    for line in f:
        ex = json.loads(line)
        ex_id = str(ex.get("example_id"))
        q = ex.get("question_text", "")
        doc = ex.get("document_text", "")
        tokens = doc.split()
        doc_trunc = " ".join(tokens[:400])
        text = q + " [SEP] " + doc_trunc
        test_texts.append(text)

        cands = ex.get("long_answer_candidates", [])
        test_meta.append((ex_id, cands, len(tokens)))

batch_size = 512
n_test = len(test_texts)
long_probs = []
short_probs = []

for start in range(0, n_test, batch_size):
    end = min(start + batch_size, n_test)
    X_batch = vectorizer.transform(test_texts[start:end])
    long_batch_prob = clf_long.predict_proba(X_batch)[:, 1]
    short_batch_prob = clf_short.predict_proba(X_batch)[:, 1]
    long_probs.extend(long_batch_prob.tolist())
    short_probs.extend(short_batch_prob.tolist())

prediction_strings = [""] * len(sample_sub)

for i, (ex_id, cands, doc_len) in enumerate(test_meta):
    if ex_id not in id_to_rows:
        continue
    row_map = id_to_rows[ex_id]

    has_long = long_probs[i] >= thr_long_final
    if has_long and "long" in row_map and len(cands) > 0:
        cand = None
        for c in cands:
            if c.get("top_level", False):
                cand = c
                break
        if cand is None:
            cand = cands[0]
        st = cand.get("start_token", 0)
        en = cand.get("end_token", 0)
        if st is None or en is None or st < 0 or en <= st or en > doc_len:
            st = 0
            en = min(10, doc_len)
        long_ps = f"{st}:{en}"
    else:
        long_ps = ""
    if "long" in row_map:
        prediction_strings[row_map["long"]] = long_ps

    has_short = short_probs[i] >= thr_short_final
    if has_short and "short" in row_map and len(cands) > 0 and long_ps != "":
        cand = None
        for c in cands:
            if c.get("top_level", False):
                cand = c
                break
        if cand is None:
            cand = cands[0]
        st = cand.get("start_token", 0)
        en = st + 1
        if st < 0 or en > doc_len:
            st = 0
            en = min(1, doc_len)
        short_ps = f"{st}:{en}"
    else:
        short_ps = ""
    if "short" in row_map:
        prediction_strings[row_map["short"]] = short_ps

sample_sub["PredictionString"] = prediction_strings

submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
sample_sub.to_csv(submission_path, index=False)
print(f"Saved submission to {submission_path}")
