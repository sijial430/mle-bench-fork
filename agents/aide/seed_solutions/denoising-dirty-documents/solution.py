import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TRAIN_CLEAN_DIR = os.path.join(INPUT_DIR, "train_cleaned")
TEST_DIR = os.path.join(INPUT_DIR, "test")

# Load training pairs: dirty -> clean pixel mapping
# Key insight: exploit the 8 shared backgrounds + simple thresholding
train_imgs = sorted([f for f in os.listdir(TRAIN_DIR) if f.endswith(".png")])
clean_imgs = sorted([f for f in os.listdir(TRAIN_CLEAN_DIR) if f.endswith(".png")])

# Build pixel-level mapping: collect (dirty_val -> clean_val) pairs
dirty_vals = []
clean_vals = []
for img_name in train_imgs:
    dirty = np.array(Image.open(os.path.join(TRAIN_DIR, img_name)).convert("L")) / 255.0
    clean_name = img_name
    clean_path = os.path.join(TRAIN_CLEAN_DIR, clean_name)
    if os.path.exists(clean_path):
        clean = np.array(Image.open(clean_path).convert("L")) / 255.0
        dirty_vals.append(dirty.flatten())
        clean_vals.append(clean.flatten())

dirty_vals = np.concatenate(dirty_vals)
clean_vals = np.concatenate(clean_vals)

# Simple approach: bin dirty values and map to mean clean value per bin
# This exploits the shared background structure
n_bins = 256
bin_edges = np.linspace(0, 1, n_bins + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_indices = np.digitize(dirty_vals, bin_edges) - 1
bin_indices = np.clip(bin_indices, 0, n_bins - 1)

bin_means = np.zeros(n_bins)
for b in range(n_bins):
    mask = bin_indices == b
    if mask.sum() > 0:
        bin_means[b] = clean_vals[mask].mean()
    else:
        bin_means[b] = bin_centers[b]

# Evaluate on training set
train_preds = bin_means[np.clip(np.digitize(dirty_vals, bin_edges) - 1, 0, n_bins - 1)]
train_rmse = np.sqrt(mean_squared_error(clean_vals, train_preds))
print(f"Training RMSE (pixel mapping): {train_rmse:.4f}")

# Generate test predictions
test_imgs = sorted([f for f in os.listdir(TEST_DIR) if f.endswith(".png")])
rows = []
for img_name in test_imgs:
    img = np.array(Image.open(os.path.join(TEST_DIR, img_name)).convert("L")) / 255.0
    h, w = img.shape
    flat = img.flatten()
    bin_idx = np.clip(np.digitize(flat, bin_edges) - 1, 0, n_bins - 1)
    predicted = bin_means[bin_idx]
    stem = os.path.splitext(img_name)[0]
    for r in range(h):
        for c in range(w):
            pixel_id = f"{stem}_{r+1}_{c+1}"
            rows.append({"id": pixel_id, "value": predicted[r * w + c]})

submission = pd.DataFrame(rows)
submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)
print(f"Saved submission with {len(submission)} rows")
