import os
import re
import zipfile
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

# Extract zipped audio files
TRAIN_DIR = os.path.join(WORKING_DIR, "train2")
TEST_DIR = os.path.join(WORKING_DIR, "test2")

for name, out_dir in [("train2.zip", TRAIN_DIR), ("test2.zip", TEST_DIR)]:
    zip_path = os.path.join(INPUT_DIR, name)
    if os.path.exists(zip_path) and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(WORKING_DIR)

# Find actual audio directories (may be nested)
if not os.listdir(TRAIN_DIR) if os.path.exists(TRAIN_DIR) else True:
    for d in os.listdir(WORKING_DIR):
        p = os.path.join(WORKING_DIR, d)
        if os.path.isdir(p) and d.startswith("train"):
            TRAIN_DIR = p
            break

if not os.path.exists(TEST_DIR) or not os.listdir(TEST_DIR):
    for d in os.listdir(WORKING_DIR):
        p = os.path.join(WORKING_DIR, d)
        if os.path.isdir(p) and d.startswith("test"):
            TEST_DIR = p
            break

# Extract spectrogram features from audio using scipy
try:
    import aifc
    import struct

    def read_aif(path):
        try:
            with aifc.open(path, "r") as f:
                n_frames = f.getnframes()
                n_channels = f.getnchannels()
                sampwidth = f.getsampwidth()
                raw = f.readframes(n_frames)
                if sampwidth == 2:
                    data = np.array(struct.unpack(f">{n_frames * n_channels}h", raw), dtype=np.float32)
                elif sampwidth == 1:
                    data = np.array(struct.unpack(f">{n_frames * n_channels}b", raw), dtype=np.float32)
                else:
                    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                return data
        except Exception:
            return np.zeros(4000)
except ImportError:
    def read_aif(path):
        return np.zeros(4000)

from scipy.signal import spectrogram as scipy_spectrogram

def extract_features(audio, sr=2000):
    if len(audio) == 0:
        audio = np.zeros(4000)
    # Compute spectrogram (top approach: contrast-enhanced spectrograms)
    f, t, Sxx = scipy_spectrogram(audio, fs=sr, nperseg=256, noverlap=128)
    Sxx = np.log1p(Sxx + 1e-10)
    # Extract statistical features from spectrogram
    feats = []
    feats.append(Sxx.mean())
    feats.append(Sxx.std())
    feats.append(Sxx.max())
    # Frequency band means (whale calls: 60-250 Hz)
    for fmin, fmax in [(0, 100), (60, 250), (100, 500), (250, 1000)]:
        mask = (f >= fmin) & (f < fmax)
        if mask.sum() > 0:
            band = Sxx[mask]
            feats.extend([band.mean(), band.std(), band.max()])
        else:
            feats.extend([0, 0, 0])
    # Temporal features
    for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
        feats.append(np.percentile(Sxx, q * 100))
    # Energy features
    feats.append(np.sum(audio ** 2))
    feats.append(np.max(np.abs(audio)))
    return np.array(feats)

# Process training data
print("Processing training audio...")
train_files = sorted([f for f in os.listdir(TRAIN_DIR) if f.endswith(".aif")])
X_train_list, y_train_list = [], []
for f in train_files:
    audio = read_aif(os.path.join(TRAIN_DIR, f))
    features = extract_features(audio)
    X_train_list.append(features)
    label = 1 if f.rstrip(".aif").endswith("_1") else 0
    y_train_list.append(label)

X_train = np.array(X_train_list)
y_train = np.array(y_train_list)
print(f"Train: {len(X_train)} samples, {y_train.sum()} positive, {len(X_train) - y_train.sum()} negative")

# Process test data
print("Processing test audio...")
test_files = sorted([f for f in os.listdir(TEST_DIR) if f.endswith(".aif")])
X_test_list, test_fnames = [], []
for f in test_files:
    audio = read_aif(os.path.join(TEST_DIR, f))
    features = extract_features(audio)
    X_test_list.append(features)
    test_fnames.append(f)

X_test = np.array(X_test_list)

# Gradient Boosting classifier (1st place approach: template matching + GBM)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(y_train))
test_preds = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_train), 1):
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
    clf.fit(X_train[tr_idx], y_train[tr_idx])
    oof_preds[val_idx] = clf.predict_proba(X_train[val_idx])[:, 1]
    test_preds += clf.predict_proba(X_test)[:, 1] / kf.n_splits
    print(f"Fold {fold} AUC: {roc_auc_score(y_train[val_idx], oof_preds[val_idx]):.4f}")

print(f"OOF AUC: {roc_auc_score(y_train, oof_preds):.4f}")

submission = pd.DataFrame({"clip": test_fnames, "probability": test_preds})
submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)
print(f"Saved submission with {len(submission)} rows")
