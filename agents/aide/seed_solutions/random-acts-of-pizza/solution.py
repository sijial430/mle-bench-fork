import os
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
import warnings

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

with open(os.path.join(INPUT_DIR, "train.json")) as f:
    train_data = json.load(f)
with open(os.path.join(INPUT_DIR, "test.json")) as f:
    test_data = json.load(f)

train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

y = train_df["requester_received_pizza"].astype(int).values

# Feature engineering (from top solution approaches)
def extract_features(df):
    feats = pd.DataFrame()
    feats["text_len"] = df.get("request_text_edit_aware", df.get("request_text", pd.Series([""] * len(df)))).fillna("").str.len()
    feats["title_len"] = df.get("request_title", pd.Series([""] * len(df))).fillna("").str.len()
    feats["account_age"] = df.get("requester_account_age_in_days_at_request", pd.Series([0] * len(df))).fillna(0)
    feats["karma_comment"] = df.get("requester_upvotes_minus_downvotes_at_request", pd.Series([0] * len(df))).fillna(0)
    feats["karma_post"] = df.get("requester_upvotes_plus_downvotes_at_request", pd.Series([0] * len(df))).fillna(0)
    feats["num_comments"] = df.get("requester_number_of_comments_at_request", pd.Series([0] * len(df))).fillna(0)
    feats["num_posts"] = df.get("requester_number_of_posts_at_request", pd.Series([0] * len(df))).fillna(0)
    feats["num_subreddits"] = df.get("requester_number_of_subreddits_at_request", pd.Series([0] * len(df))).fillna(0)
    feats["raop_posts"] = df.get("requester_number_of_posts_on_raop_at_request", pd.Series([0] * len(df))).fillna(0)
    feats["raop_comments"] = df.get("requester_number_of_comments_in_raop_at_request", pd.Series([0] * len(df))).fillna(0)
    return feats.fillna(0)

train_feats = extract_features(train_df)
test_feats = extract_features(test_df)

scaler = StandardScaler()
X_train_meta = csr_matrix(scaler.fit_transform(train_feats.values))
X_test_meta = csr_matrix(scaler.transform(test_feats.values))

# TF-IDF on request text
text_col = "request_text_edit_aware" if "request_text_edit_aware" in train_df.columns else "request_text"
train_text = train_df.get(text_col, pd.Series([""] * len(train_df))).fillna("")
test_text = test_df.get(text_col, pd.Series([""] * len(test_df))).fillna("")

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
tfidf.fit(pd.concat([train_text, test_text]))
X_train_text = tfidf.transform(train_text)
X_test_text = tfidf.transform(test_text)

X_train = hstack([X_train_meta, X_train_text]).tocsr()
X_test = hstack([X_test_meta, X_test_text]).tocsr()

# 5-fold CV with LR
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train, y), 1):
    clf = LogisticRegression(C=1.0, solver="liblinear", max_iter=1000)
    clf.fit(X_train[tr_idx], y[tr_idx])
    oof_preds[val_idx] = clf.predict_proba(X_train[val_idx])[:, 1]
    test_preds += clf.predict_proba(X_test)[:, 1] / kf.n_splits
    print(f"Fold {fold} AUC: {roc_auc_score(y[val_idx], oof_preds[val_idx]):.4f}")

print(f"OOF AUC: {roc_auc_score(y, oof_preds):.4f}")

submission = pd.DataFrame({
    "request_id": test_df["request_id"],
    "requester_received_pizza": test_preds,
})
submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)
print(f"Saved submission with {len(submission)} rows")
