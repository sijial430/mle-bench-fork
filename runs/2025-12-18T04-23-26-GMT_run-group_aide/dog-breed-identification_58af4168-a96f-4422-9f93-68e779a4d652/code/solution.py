import os
import sys
import time
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

# Config
RANDOM_SEED = 42
BATCH_SIZE = 64
IMG_SIZE = 224
NUM_WORKERS = 0  # 0 for compatibility in restricted environments
N_SPLITS = 5

TRAIN_DIR = "./input/train"
TEST_DIR = "./input/test"
LABELS_CSV = "./input/labels.csv"
SAMPLE_SUB = "./input/sample_submission.csv"
SUBMISSION_DIR = "./submission"
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

os.makedirs(SUBMISSION_DIR, exist_ok=True)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Read data
labels_df = pd.read_csv(LABELS_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)

train_ids = labels_df["id"].values
train_breeds = labels_df["breed"].values
test_df = pd.read_csv(SAMPLE_SUB)[["id"]]

# Label encode breeds
le = LabelEncoder()
y_all = le.fit_transform(train_breeds)
class_names = list(le.classes_)
num_classes = len(class_names)
print(
    f"Found {len(train_ids)} train images, {len(test_df)} test images, {num_classes} classes."
)

# Image dataset for feature extraction
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class ImagePathDataset(Dataset):
    def __init__(self, ids, img_dir, transform=None):
        self.ids = ids
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            # If failed to load, return a black image
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
        if self.transform:
            img = self.transform(img)
        return img_id, img


# Build pretrained model for feature extraction
device = torch.device("cpu")
backbone = models.resnet18(pretrained=True)


# Remove final classifier
class Identity(nn.Module):
    def forward(self, x):
        return x


backbone.fc = Identity()
backbone.eval()
backbone.to(device)


def extract_features(ids_list, img_dir, batch_size=BATCH_SIZE):
    ds = ImagePathDataset(ids_list, img_dir, transform=transform)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS
    )
    features = []
    ids_out = []
    with torch.no_grad():
        for batch in tqdm(
            loader, desc=f"Extracting features from {os.path.basename(img_dir)}"
        ):
            batch_ids, imgs = batch
            imgs = imgs.to(device)
            feats = backbone(imgs)
            # If backbone outputs 512 x 1 x 1 sometimes, flatten
            feats = feats.view(feats.size(0), -1).cpu().numpy()
            features.append(feats)
            ids_out.extend(batch_ids)
    features = np.vstack(features).astype(np.float32)
    return ids_out, features


# Precompute features for all train images (in the same order as train_ids)
print("Extracting train features...")
train_ids_list = train_ids.tolist()
ids_ordered, X_train = extract_features(train_ids_list, TRAIN_DIR)
# ensure ordering matches
if ids_ordered != train_ids_list:
    # reorder to match original
    id_to_idx = {idv: i for i, idv in enumerate(ids_ordered)}
    order = [id_to_idx[idv] for idv in train_ids_list]
    X_train = X_train[order]

print("Extracting test features...")
test_ids_list = test_df["id"].values.tolist()
ids_test_ordered, X_test = extract_features(test_ids_list, TEST_DIR)
if ids_test_ordered != test_ids_list:
    id_to_idx = {idv: i for i, idv in enumerate(ids_test_ordered)}
    order = [id_to_idx[idv] for idv in test_ids_list]
    X_test = X_test[order]

# 5-fold stratified CV with Logistic Regression on extracted features
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

oof_probs = np.zeros((len(train_ids_list), num_classes), dtype=np.float64)
test_probs = np.zeros((len(test_ids_list), num_classes), dtype=np.float64)

fold = 0
for train_idx, val_idx in skf.split(X_train, y_all):
    fold += 1
    print(f"\nFold {fold}/{N_SPLITS} -- Train {len(train_idx)} val {len(val_idx)}")
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_all[train_idx], y_all[val_idx]

    # Scale features modestly (mean centering)
    # Using simple standardization can help logistic regression
    mean = X_tr.mean(axis=0, keepdims=True)
    std = X_tr.std(axis=0, keepdims=True) + 1e-6
    X_tr_s = (X_tr - mean) / std
    X_val_s = (X_val - mean) / std
    X_test_s = (X_test - mean) / std

    # Train Logistic Regression
    clf = LogisticRegression(
        multi_class="multinomial",
        solver="saga",
        max_iter=1000,
        C=1.0,
        random_state=RANDOM_SEED,
        n_jobs=1,
    )
    clf.fit(X_tr_s, y_tr)

    # Predict
    val_pred = clf.predict_proba(X_val_s)
    test_pred = clf.predict_proba(X_test_s)

    oof_probs[val_idx] = val_pred
    test_probs += test_pred

    # Fold log loss
    fold_loss = log_loss(y_val, val_pred, labels=list(range(num_classes)))
    print(f"Fold {fold} log_loss: {fold_loss:.5f}")

# Average test probabilities
test_probs /= N_SPLITS

# Overall CV log loss
cv_logloss = log_loss(y_all, oof_probs, labels=list(range(num_classes)))
print(f"\nOverall CV log_loss (5-fold): {cv_logloss:.5f}")

# Prepare submission in the order of sample_submission columns
submission = pd.DataFrame(test_probs, columns=le.classes_)
submission.insert(0, "id", test_ids_list)

# Ensure sample_submission order (some competitions require exact column order)
sample_cols = list(pd.read_csv(SAMPLE_SUB).columns)
# If sample has same set of columns, reorder
if set(sample_cols) == set(list(submission.columns)):
    submission = submission[sample_cols]
else:
    # Ensure 'id' first and then our class order
    cols = ["id"] + [c for c in submission.columns if c != "id"]
    submission = submission[cols]

submission.to_csv(SUBMISSION_FILE, index=False)
print(f"Saved submission to {SUBMISSION_FILE}")

# Print a small sample of submission
print("\nSubmission sample rows:")
print(submission.head(3))

# Final print of metric
print(f"\nFinal reported CV log_loss: {cv_logloss:.6f}")
