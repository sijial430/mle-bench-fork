import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

# Load data structures
cv_folds = pd.read_csv(os.path.join(INPUT_DIR, "essential_data/CVfolds_2.txt"))
rec_id2filename = pd.read_csv(os.path.join(INPUT_DIR, "essential_data/rec_id2filename.txt"))
sample_sub = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv"))

# Parse labels
labels_path = os.path.join(INPUT_DIR, "essential_data/rec_labels_test_hidden.txt")
with open(labels_path) as f:
    lines = f.read().strip().split("\n")
header = lines[0]
rec_labels = {}
for line in lines[1:]:
    parts = line.split(",")
    rec_id = int(parts[0])
    if len(parts) > 1 and parts[1] != "?":
        species = [int(x) for x in parts[1:] if x.strip()]
        rec_labels[rec_id] = species
    else:
        rec_labels[rec_id] = None

# Load histogram of segments as features
hist_path = os.path.join(INPUT_DIR, "supplemental_data/histogram_of_segments.txt")
hist_data = {}
with open(hist_path) as f:
    lines = f.read().strip().split("\n")
for line in lines[1:]:
    parts = line.split(",")
    rec_id = int(parts[0])
    features = [float(x) for x in parts[1:]]
    hist_data[rec_id] = features

# Identify train and test rec_ids from cv_folds
train_recs = cv_folds[cv_folds["fold"] == 0]["rec_id"].values
test_recs = cv_folds[cv_folds["fold"] == 1]["rec_id"].values

NUM_SPECIES = 19

# Build feature matrices
all_recs = sorted(hist_data.keys())
feat_dim = len(next(iter(hist_data.values())))

X_train_list, y_train_list = [], []
for rec_id in train_recs:
    if rec_id in hist_data and rec_labels.get(rec_id) is not None:
        X_train_list.append(hist_data[rec_id])
        label_vec = np.zeros(NUM_SPECIES)
        for sp in rec_labels[rec_id]:
            if 0 <= sp < NUM_SPECIES:
                label_vec[sp] = 1
        y_train_list.append(label_vec)

X_train = np.array(X_train_list)
y_train = np.array(y_train_list)

X_test_list, test_rec_order = [], []
for rec_id in test_recs:
    if rec_id in hist_data:
        X_test_list.append(hist_data[rec_id])
        test_rec_order.append(rec_id)

X_test = np.array(X_test_list) if X_test_list else np.zeros((len(test_recs), feat_dim))
if not X_test_list:
    test_rec_order = list(test_recs)

# Binary Relevance: one classifier per species (1st place: template matching + gradient boosting)
test_preds = np.zeros((len(test_rec_order), NUM_SPECIES))
for sp in range(NUM_SPECIES):
    y_sp = y_train[:, sp]
    if y_sp.sum() == 0 or y_sp.sum() == len(y_sp):
        test_preds[:, sp] = y_sp.mean()
        continue
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    clf.fit(X_train, y_sp)
    test_preds[:, sp] = clf.predict_proba(X_test)[:, 1]
    print(f"Species {sp}: train AUC = {roc_auc_score(y_sp, clf.predict_proba(X_train)[:, 1]):.3f}")

# Build submission: Id = rec_id * 100 + species_id
ids, probs = [], []
for i, rec_id in enumerate(test_rec_order):
    for sp in range(NUM_SPECIES):
        ids.append(rec_id * 100 + sp)
        probs.append(test_preds[i, sp])

submission = pd.DataFrame({"Id": ids, "Probability": probs})
submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)
print(f"Saved submission with {len(submission)} rows")
