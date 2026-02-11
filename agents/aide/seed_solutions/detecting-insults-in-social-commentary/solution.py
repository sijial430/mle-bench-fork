import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack
import warnings

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

# Load data
train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))

train["Comment"] = train["Comment"].fillna("")
test["Comment"] = test["Comment"].fillna("")

y = train["Insult"].values

# TF-IDF: word + character n-grams (top approach from 1st place)
word_vec = TfidfVectorizer(
    sublinear_tf=True, strip_accents="unicode", analyzer="word",
    token_pattern=r"\w{1,}", ngram_range=(1, 2), max_features=50000,
)
char_vec = TfidfVectorizer(
    sublinear_tf=True, strip_accents="unicode", analyzer="char",
    ngram_range=(2, 5), max_features=50000,
)

all_text = pd.concat([train["Comment"], test["Comment"]])
word_vec.fit(all_text)
char_vec.fit(all_text)

X_train = hstack([word_vec.transform(train["Comment"]), char_vec.transform(train["Comment"])]).tocsr()
X_test = hstack([word_vec.transform(test["Comment"]), char_vec.transform(test["Comment"])]).tocsr()

# 5-fold CV with LR ensemble (SVC-like approach from 1st place)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train, y), 1):
    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    for C in [1.0, 5.0]:
        clf = LogisticRegression(C=C, solver="liblinear", max_iter=1000)
        clf.fit(X_tr, y_tr)
        oof_preds[val_idx] += clf.predict_proba(X_val)[:, 1] / 2
        test_preds += clf.predict_proba(X_test)[:, 1] / (2 * kf.n_splits)
    print(f"Fold {fold} AUC: {roc_auc_score(y_val, oof_preds[val_idx]):.4f}")

print(f"OOF AUC: {roc_auc_score(y, oof_preds):.4f}")

# Submission format: Insult, Date, Comment (matched by Comment)
submission = pd.DataFrame({
    "Insult": test_preds,
    "Date": test["Date"],
    "Comment": test["Comment"],
})
submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)
print(f"Saved submission with {len(submission)} rows")
