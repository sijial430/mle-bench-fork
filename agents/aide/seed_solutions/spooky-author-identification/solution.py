import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from scipy.sparse import hstack
import warnings

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))

authors = ["EAP", "HPL", "MWS"]
label_map = {a: i for i, a in enumerate(authors)}
y = train["author"].map(label_map).values

# Character n-gram TF-IDF + word TF-IDF (dominant winning approach)
word_vec = TfidfVectorizer(
    sublinear_tf=True, strip_accents="unicode", analyzer="word",
    token_pattern=r"\w{1,}", ngram_range=(1, 2), max_features=50000,
)
char_vec = TfidfVectorizer(
    sublinear_tf=True, strip_accents="unicode", analyzer="char",
    ngram_range=(2, 6), max_features=50000,
)

all_text = pd.concat([train["text"], test["text"]])
word_vec.fit(all_text)
char_vec.fit(all_text)

X_train = hstack([word_vec.transform(train["text"]), char_vec.transform(train["text"])]).tocsr()
X_test = hstack([word_vec.transform(test["text"]), char_vec.transform(test["text"])]).tocsr()

# 5-fold CV
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((len(train), 3))
test_preds = np.zeros((len(test), 3))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train, y), 1):
    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    clf = LogisticRegression(C=1.0, solver="lbfgs", multi_class="multinomial", max_iter=1000)
    clf.fit(X_tr, y_tr)
    oof_preds[val_idx] = clf.predict_proba(X_val)
    test_preds += clf.predict_proba(X_test) / kf.n_splits
    print(f"Fold {fold} log_loss: {log_loss(y_val, oof_preds[val_idx]):.4f}")

print(f"OOF log_loss: {log_loss(y, oof_preds):.4f}")

submission = pd.DataFrame({"id": test["id"]})
for i, author in enumerate(authors):
    submission[author] = test_preds[:, i]
submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)
print(f"Saved submission with {len(submission)} rows")
