import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import warnings

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))

# Encode species labels
le = LabelEncoder()
y = le.fit_transform(train["species"])
species_names = le.classes_

# Features: all columns except id and species
feature_cols = [c for c in train.columns if c not in ["id", "species"]]
X_train = train[feature_cols].values
X_test = test[feature_cols].values

# StandardScaler (key preprocessing step from 1st place)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression (dominant winning approach)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
n_classes = len(species_names)
oof_preds = np.zeros((len(train), n_classes))
test_preds = np.zeros((len(test), n_classes))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train, y), 1):
    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    clf = LogisticRegression(C=1000, solver="lbfgs", multi_class="multinomial", max_iter=2000)
    clf.fit(X_tr, y_tr)
    oof_preds[val_idx] = clf.predict_proba(X_val)
    test_preds += clf.predict_proba(X_test) / kf.n_splits
    print(f"Fold {fold} log_loss: {log_loss(y_val, oof_preds[val_idx]):.4f}")

print(f"OOF log_loss: {log_loss(y, oof_preds):.4f}")

submission = pd.DataFrame({"id": test["id"]})
for i, species in enumerate(species_names):
    submission[species] = test_preds[:, i]
submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)
print(f"Saved submission with {len(submission)} rows")
