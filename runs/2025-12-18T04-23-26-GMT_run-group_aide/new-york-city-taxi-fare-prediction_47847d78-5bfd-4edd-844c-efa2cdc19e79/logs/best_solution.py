import os
import sys
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

# Settings
INPUT_DIR = "./input"
TRAIN_FILE = os.path.join(INPUT_DIR, "labels.csv")
TEST_FILE = os.path.join(INPUT_DIR, "test.csv")
SUBMISSION_DIR = "./submission"
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")
SAMPLE_NROWS = 200_000  # sample size to fit within runtime/memory
RANDOM_STATE = 42
N_FOLDS = 5

os.makedirs(SUBMISSION_DIR, exist_ok=True)


def haversine_vectorized(lat1, lon1, lat2, lon2):
    # convert degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km


def preprocess(df, is_train=True):
    # Parse datetime
    df = df.copy()
    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"], utc=True, errors="coerce"
    )
    # Extract datetime features
    df["pickup_hour"] = df["pickup_datetime"].dt.hour
    df["pickup_day"] = df["pickup_datetime"].dt.day
    df["pickup_month"] = df["pickup_datetime"].dt.month
    df["pickup_weekday"] = df["pickup_datetime"].dt.weekday
    df["pickup_minute"] = df["pickup_datetime"].dt.minute
    df["is_weekend"] = df["pickup_weekday"].isin([5, 6]).astype(int)
    # Distance features
    df["haversine_km"] = haversine_vectorized(
        df["pickup_latitude"].values,
        df["pickup_longitude"].values,
        df["dropoff_latitude"].values,
        df["dropoff_longitude"].values,
    )
    # Simple manhattan on degrees (not used for distance measure but as feature)
    df["manhattan_deg"] = np.abs(
        df["pickup_latitude"] - df["dropoff_latitude"]
    ) + np.abs(df["pickup_longitude"] - df["dropoff_longitude"])
    # basic passenger_count: replace 0s in test or bad values with 1
    df["passenger_count"] = df["passenger_count"].fillna(1).astype(int)
    df.loc[df["passenger_count"] < 1, "passenger_count"] = 1
    # Keep useful columns
    features = [
        "haversine_km",
        "manhattan_deg",
        "pickup_hour",
        "pickup_weekday",
        "pickup_month",
        "pickup_minute",
        "is_weekend",
        "passenger_count",
    ]
    if is_train:
        return df[features + ["fare_amount"]].copy()
    else:
        return df[features].copy()


# Load a manageable sample of training data
print("Loading training sample...")
try:
    train = pd.read_csv(TRAIN_FILE, nrows=SAMPLE_NROWS)
except Exception as e:
    print("Failed to read training file with nrows sample:", e)
    try:
        train = pd.read_csv(TRAIN_FILE)
    except Exception as e2:
        print("Failed to read training file without nrows:", e2)
        sys.exit(1)

# Basic cleaning: drop rows with missing coords/datetimes
train = train.dropna(
    subset=[
        "pickup_latitude",
        "pickup_longitude",
        "dropoff_latitude",
        "dropoff_longitude",
        "pickup_datetime",
    ]
)
# Filter to reasonable NYC bounding box to remove corrupted rows (relatively tight)
lat_min, lat_max = 40.0, 42.0
lon_min, lon_max = -75.0, -72.0
train = train[
    (train["pickup_latitude"].between(lat_min, lat_max))
    & (train["dropoff_latitude"].between(lat_min, lat_max))
    & (train["pickup_longitude"].between(lon_min, lon_max))
    & (train["dropoff_longitude"].between(lon_min, lon_max))
]
# Filter fares to reasonable range
train = train[train["fare_amount"].between(0, 500)]
# Filter passenger_count to reasonable values
train = train[train["passenger_count"].between(1, 6)]

if len(train) < 1000:
    print(
        "Too few rows after filtering; loading more rows without nrows limit (slower)."
    )
    train = pd.read_csv(TRAIN_FILE)
    train = train.dropna(
        subset=[
            "pickup_latitude",
            "pickup_longitude",
            "dropoff_latitude",
            "dropoff_longitude",
            "pickup_datetime",
        ]
    )
    train = train[
        (train["pickup_latitude"].between(lat_min, lat_max))
        & (train["dropoff_latitude"].between(lat_min, lat_max))
        & (train["pickup_longitude"].between(lon_min, lon_max))
        & (train["dropoff_longitude"].between(lon_min, lon_max))
    ]
    train = train[train["fare_amount"].between(0, 500)]
    train = train[train["passenger_count"].between(1, 6)]
    # sample down if too large
    train = train.sample(n=min(SAMPLE_NROWS, len(train)), random_state=RANDOM_STATE)

print(f"Using {len(train)} rows for training/validation.")

# Preprocess
train_proc = preprocess(train, is_train=True)
X = train_proc.drop(columns=["fare_amount"])
y = train_proc["fare_amount"].values

# 5-fold cross-validation using sklearn API of LightGBM to support early stopping
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
rmse_scores = []
models = []
best_iters = []
fold = 0

for train_idx, val_idx in kf.split(X):
    fold += 1
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    print(
        f"Training fold {fold} with {len(X_train)} train rows and {len(X_val)} val rows..."
    )

    lgb_model = lgb.LGBMRegressor(
        objective="regression",
        learning_rate=0.1,
        n_estimators=1000,
        num_leaves=31,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        silent=True,
    )
    # Fit with early stopping using sklearn API
    try:
        lgb_model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            early_stopping_rounds=50,
            verbose=False,
        )
    except TypeError:
        # In case older lightgbm versions don't support early_stopping_rounds in sklearn API
        # fallback to using callbacks
        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
        lgb_model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=callbacks,
        )

    # Predict and evaluate
    val_pred = lgb_model.predict(
        X_val, num_iteration=getattr(lgb_model, "best_iteration_", None)
    )
    rmse = math.sqrt(mean_squared_error(y_val, val_pred))
    print(f"Fold {fold} RMSE: {rmse:.5f}")
    rmse_scores.append(rmse)
    models.append(lgb_model)
    bi = getattr(lgb_model, "best_iteration_", None)
    if bi is None or bi == 0:
        bi = lgb_model.n_estimators
    best_iters.append(bi)

mean_rmse = float(np.mean(rmse_scores))
std_rmse = float(np.std(rmse_scores))
print(f"CV mean RMSE: {mean_rmse:.5f}  std: {std_rmse:.5f}")

# Train final model on all sample data using averaged number of iterations
avg_best_iter = max(1, int(np.mean(best_iters)))
print(f"Retraining final model on all sampled data for {avg_best_iter} rounds...")
final_model = lgb.LGBMRegressor(
    objective="regression",
    learning_rate=0.1,
    n_estimators=avg_best_iter,
    num_leaves=31,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    silent=True,
)
final_model.fit(X, y)

# Load test data and preprocess
print("Loading and preprocessing test set...")
test = pd.read_csv(TEST_FILE)
test_ids = test["key"].values
test_proc = preprocess(test, is_train=False)

# Predict
preds = final_model.predict(test_proc)
# Ensure no negative predictions
preds = np.clip(preds, 0, None)

# Prepare submission
submission = pd.DataFrame({"key": test_ids, "fare_amount": preds})
submission.to_csv(SUBMISSION_FILE, index=False)
print(f"Saved submission to {SUBMISSION_FILE}")

# Print final reported metric
print(f"Final reported CV RMSE: {mean_rmse:.6f}")
