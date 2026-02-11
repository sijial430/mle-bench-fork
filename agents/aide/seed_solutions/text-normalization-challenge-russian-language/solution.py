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
for zf_name in ["ru_train.csv.zip", "ru_test_2.csv.zip", "ru_sample_submission_2.csv.zip"]:
    zf_path = os.path.join(INPUT_DIR, zf_name)
    if os.path.exists(zf_path):
        with zipfile.ZipFile(zf_path, "r") as zf:
            zf.extractall(WORKING_DIR)

# Load data
train_path = os.path.join(WORKING_DIR, "ru_train.csv")
if not os.path.exists(train_path):
    train_path = os.path.join(INPUT_DIR, "ru_train.csv")
test_path = os.path.join(WORKING_DIR, "ru_test_2.csv")
if not os.path.exists(test_path):
    test_path = os.path.join(INPUT_DIR, "ru_test_2.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print(f"Training data: {len(train)} tokens")
print(f"Test data: {len(test)} tokens")

# Build lookup table: before -> most common after (rule-based approach)
# Top approach: seq2seq NMT (1st place, University of Stuttgart)
# Practical approach: lookup table from training data
token_mapping = {}
for _, row in train.iterrows():
    before = str(row["before"])
    after = str(row["after"])
    if before not in token_mapping:
        token_mapping[before] = {}
    token_mapping[before][after] = token_mapping[before].get(after, 0) + 1

best_mapping = {}
for before, after_counts in token_mapping.items():
    best_after = max(after_counts, key=after_counts.get)
    best_mapping[before] = best_after

def normalize_token(before):
    before_str = str(before)
    if before_str in best_mapping:
        return best_mapping[before_str]
    # Cyrillic letters: return as-is
    if re.match(r'^[\u0400-\u04FF]+$', before_str):
        return before_str
    # Digits: return as-is
    if re.match(r'^\d+$', before_str):
        return before_str
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

# Validate format against sample submission
sample_path = os.path.join(WORKING_DIR, "ru_sample_submission_2.csv")
if not os.path.exists(sample_path):
    sample_path = os.path.join(INPUT_DIR, "ru_sample_submission_2.csv")
if os.path.exists(sample_path):
    sample_sub = pd.read_csv(sample_path)
    submission = submission[submission["id"].isin(sample_sub["id"])]
    missing = set(sample_sub["id"]) - set(submission["id"])
    if missing:
        missing_df = pd.DataFrame({"id": list(missing), "after": ""})
        submission = pd.concat([submission, missing_df])
    submission = submission.sort_values("id").reset_index(drop=True)

# Compute validation accuracy
val_sample = train.sample(min(10000, len(train)), random_state=42)
correct = sum(1 for _, row in val_sample.iterrows() if str(normalize_token(str(row["before"]))) == str(row["after"]))
val_acc = correct / len(val_sample)
print(f"Validation accuracy (sample): {val_acc:.4f}")

submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)
print(f"Saved submission with {len(submission)} rows")
