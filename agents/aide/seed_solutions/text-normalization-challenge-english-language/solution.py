import os
import re
import zipfile
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

# Extract zipped data
for zf_name in ["en_train.csv.zip", "en_test_2.csv.zip", "en_sample_submission_2.csv.zip"]:
    zf_path = os.path.join(INPUT_DIR, zf_name)
    if os.path.exists(zf_path):
        with zipfile.ZipFile(zf_path, "r") as zf:
            zf.extractall(WORKING_DIR)

# Load data
train_path = os.path.join(WORKING_DIR, "en_train.csv")
if not os.path.exists(train_path):
    train_path = os.path.join(INPUT_DIR, "en_train.csv")
test_path = os.path.join(WORKING_DIR, "en_test_2.csv")
if not os.path.exists(test_path):
    test_path = os.path.join(INPUT_DIR, "en_test_2.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# Build a rule-based normalizer (top approach: manual grammar engineering for semiotic classes)
# Learn token mappings from training data
print(f"Training data: {len(train)} tokens")
print(f"Test data: {len(test)} tokens")

# Build lookup table: (class, before) -> most common after
# Since class is not available in test, we build: before -> most common after
token_mapping = {}
for _, row in train.iterrows():
    before = str(row["before"])
    after = str(row["after"])
    if before not in token_mapping:
        token_mapping[before] = {}
    token_mapping[before][after] = token_mapping[before].get(after, 0) + 1

# For each before token, pick the most common after
best_mapping = {}
for before, after_counts in token_mapping.items():
    best_after = max(after_counts, key=after_counts.get)
    best_mapping[before] = best_after

# Rule-based normalization for common semiotic classes
def normalize_token(before):
    before_str = str(before)

    # Direct lookup
    if before_str in best_mapping:
        return best_mapping[before_str]

    # If it's just letters, return as-is (PLAIN class)
    if re.match(r'^[a-zA-Z]+$', before_str):
        return before_str

    # Numbers
    if re.match(r'^\d+$', before_str):
        return before_str  # Keep as digit string (self-normalization)

    # Default: return as-is
    return before_str

# Apply normalization
print("Normalizing test tokens...")
predictions = []
for _, row in test.iterrows():
    before = str(row["before"])
    after = normalize_token(before)
    sid = row["sentence_id"]
    tid = row["token_id"]
    predictions.append({
        "id": f"{sid}_{tid}",
        "after": after,
    })

submission = pd.DataFrame(predictions)

# Validate against sample submission format
sample_path = os.path.join(WORKING_DIR, "en_sample_submission_2.csv")
if not os.path.exists(sample_path):
    sample_path = os.path.join(INPUT_DIR, "en_sample_submission_2.csv")
if os.path.exists(sample_path):
    sample_sub = pd.read_csv(sample_path)
    # Ensure we have exactly the same IDs
    submission = submission[submission["id"].isin(sample_sub["id"])]
    missing = set(sample_sub["id"]) - set(submission["id"])
    if missing:
        missing_df = pd.DataFrame({"id": list(missing), "after": ""})
        submission = pd.concat([submission, missing_df])
    submission = submission.sort_values("id").reset_index(drop=True)

# Compute accuracy on a small validation set from training
val_sample = train.sample(min(10000, len(train)), random_state=42)
correct = 0
for _, row in val_sample.iterrows():
    pred = normalize_token(str(row["before"]))
    if str(pred) == str(row["after"]):
        correct += 1
val_acc = correct / len(val_sample)
print(f"Validation accuracy (sample): {val_acc:.4f}")

submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)
print(f"Saved submission with {len(submission)} rows")
