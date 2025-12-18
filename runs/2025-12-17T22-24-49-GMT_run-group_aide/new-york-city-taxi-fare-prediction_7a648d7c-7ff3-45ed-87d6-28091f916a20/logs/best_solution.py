import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

# Paths
INPUT_DIR = "./input"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

train_path = os.path.join(INPUT_DIR, "labels.csv")
test_path = os.path.join(INPUT_DIR, "test.csv")
sample_sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")

# Parameters
N_ROWS = 4_000_000  # subsample size
RANDOM_STATE = 42
N_FOLDS = 5
MAX_NUM_BOOST_ROUND = 2000
EARLY_STOPPING_ROUNDS = 50


def haversine_distance(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance (in km)."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # km
    return c * r


def manhattan_distance(lat1, lon1, lat2, lon2):
    a = haversine_distance(lat1, lon1, lat1, lon2)
    b = haversine_distance(lat1, lon1, lat2, lon1)
    return a + b


def bearing_array(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.degrees(np.arctan2(y, x))


def add_time_features(df):
    df = df.copy()
    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"].astype(str).str.replace(" UTC", "", regex=False),
        errors="coerce",
    )
    df["pickup_year"] = df["pickup_datetime"].dt.year
    df["pickup_month"] = df["pickup_datetime"].dt.month
    df["pickup_day"] = df["pickup_datetime"].dt.day
    df["pickup_hour"] = df["pickup_datetime"].dt.hour
    df["pickup_dow"] = df["pickup_datetime"].dt.weekday
    df["pickup_week"] = df["pickup_datetime"].dt.isocalendar().week.astype(int)

    # Extra time features
    df["is_weekend"] = df["pickup_dow"].isin([5, 6]).astype(int)
    # Rush hour heuristic: 7-9am and 16-19
    df["is_rush_hour"] = (
        df["pickup_hour"].between(7, 9) | df["pickup_hour"].between(16, 19)
    ).astype(int)
    # Sine/cosine of hour to capture circularity
    df["sin_hour"] = np.sin(2 * np.pi * df["pickup_hour"] / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * df["pickup_hour"] / 24.0)
    return df


def add_geo_features(df):
    df = df.copy()
    df["distance_haversine"] = haversine_distance(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )
    df["distance_manhattan"] = manhattan_distance(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )
    df["direction"] = bearing_array(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )

    # Extra distance-based features
    df["log_distance_haversine"] = np.log1p(df["distance_haversine"])
    df["log_distance_manhattan"] = np.log1p(df["distance_manhattan"])

    # Distance * passenger_count interaction (helps capture per-passenger effect)
    if "passenger_count" in df.columns:
        df["dist_hav_x_passenger"] = df["distance_haversine"] * df["passenger_count"]
        df["dist_man_x_passenger"] = df["distance_manhattan"] * df["passenger_count"]

    return df


def add_target_leak_safe_features(df):
    """
    Add features that depend on fare_amount but only for training data.
    Should only be called on training data with 'fare_amount' present.
    These are safe because they will not be computed for test.
    """
    df = df.copy()
    # Fare per km (proxy for speed/tolls), using both distances
    eps = 1e-3
    df["fare_per_km_hav"] = df["fare_amount"] / (df["distance_haversine"] + eps)
    df["fare_per_km_man"] = df["fare_amount"] / (df["distance_manhattan"] + eps)
    # Clip extreme values to reduce noise
    df["fare_per_km_hav"] = df["fare_per_km_hav"].clip(0, 100)
    df["fare_per_km_man"] = df["fare_per_km_man"].clip(0, 100)
    return df


def clean_data(df, is_train=True):
    df = df.copy()
    # Basic geographic bounds around NYC (with generous limits)
    mask = (
        (df["pickup_longitude"] > -80)
        & (df["pickup_longitude"] < -70)
        & (df["dropoff_longitude"] > -80)
        & (df["dropoff_longitude"] < -70)
        & (df["pickup_latitude"] > 35)
        & (df["pickup_latitude"] < 45)
        & (df["dropoff_latitude"] > 35)
        & (df["dropoff_latitude"] < 45)
    )
    df = df[mask].copy()
    df = add_time_features(df)
    df = add_geo_features(df)
    # Passenger count sanity
    df = df[(df["passenger_count"] > 0) & (df["passenger_count"] <= 6)]
    if is_train and "fare_amount" in df.columns:
        # Fare sanity
        df = df[(df["fare_amount"] > 0) & (df["fare_amount"] < 200)]
        df = add_target_leak_safe_features(df)
    return df


# Load a subsample of training data
print("Loading training data...")
train_df = pd.read_csv(train_path, nrows=N_ROWS)
print(f"Loaded {len(train_df)} rows from training file.")

# Drop rows with missing basic coordinates/passenger_count
train_df = train_df.dropna(
    subset=[
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]
)

train_df = clean_data(train_df, is_train=True)
print(f"After cleaning: {len(train_df)} training rows.")

# Define feature columns (include new features)
feature_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "pickup_year",
    "pickup_month",
    "pickup_day",
    "pickup_hour",
    "pickup_dow",
    "pickup_week",
    "is_weekend",
    "is_rush_hour",
    "sin_hour",
    "cos_hour",
    "distance_haversine",
    "distance_manhattan",
    "direction",
    "log_distance_haversine",
    "log_distance_manhattan",
    "dist_hav_x_passenger",
    "dist_man_x_passenger",
    # target-leak-safe but training-only features
    "fare_per_km_hav",
    "fare_per_km_man",
]

# Ensure no missing values in feature columns (drop rows with NaNs)
train_df = train_df.dropna(subset=feature_cols + ["fare_amount"])
X = train_df[feature_cols]
y = train_df["fare_amount"].astype(float).values

# Global mean fare for fallback
global_mean_fare = float(y.mean())

# LightGBM parameters for core API
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "verbosity": -1,
    "seed": RANDOM_STATE,
}

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
rmse_scores = []
best_iterations = []

print("Starting 5-fold cross-validation with early stopping via lightgbm.train...")
fold_num = 1
for train_idx, val_idx in kf.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

    evals_result = {}
    gbm = lgb.train(
        params=lgb_params,
        train_set=lgb_train,
        num_boost_round=MAX_NUM_BOOST_ROUND,
        valid_sets=[lgb_val],
        valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.record_evaluation(evals_result),
        ],
    )

    best_iter = gbm.best_iteration
    if best_iter is None or best_iter <= 0:
        best_iter = MAX_NUM_BOOST_ROUND
    best_iterations.append(best_iter)

    preds_val = gbm.predict(X_val, num_iteration=best_iter)
    rmse = mean_squared_error(y_val, preds_val, squared=False)
    rmse_scores.append(rmse)
    print(f"Fold {fold_num} RMSE: {rmse:.5f}, best_iteration: {best_iter}")
    fold_num += 1

cv_rmse = float(np.mean(rmse_scores))
avg_best_iteration = int(np.round(np.mean(best_iterations)))
if avg_best_iteration <= 0:
    avg_best_iteration = MAX_NUM_BOOST_ROUND
print(f"Mean CV RMSE over {N_FOLDS} folds: {cv_rmse:.5f}")
print(f"Average best iteration from CV: {avg_best_iteration}")

# Train final model on all data using averaged best iteration
print("Training final model on all data...")
lgb_train_full = lgb.Dataset(X, label=y)
final_model = lgb.train(
    params=lgb_params,
    train_set=lgb_train_full,
    num_boost_round=avg_best_iteration,
    valid_sets=[],
)

# Load and prepare test data
print("Loading test data...")
test_full = pd.read_csv(test_path)

# Add features (no hard filtering to avoid dropping rows)
test_full = add_time_features(test_full)
test_full = add_geo_features(test_full)

# For test data, we cannot compute training-only features (fare_per_km_*), so set them to sane defaults
test_full["fare_per_km_hav"] = global_mean_fare / (
    test_full["distance_haversine"] + 1e-3
)
test_full["fare_per_km_man"] = global_mean_fare / (
    test_full["distance_manhattan"] + 1e-3
)
test_full["fare_per_km_hav"] = test_full["fare_per_km_hav"].clip(0, 100)
test_full["fare_per_km_man"] = test_full["fare_per_km_man"].clip(0, 100)

# Ensure all feature columns are present
for col in feature_cols:
    if col not in test_full.columns:
        test_full[col] = np.nan

X_test = test_full[feature_cols]

print("Predicting on test data...")
test_preds = final_model.predict(X_test, num_iteration=avg_best_iteration)

# Replace any non-finite predictions with global mean
test_preds = np.where(np.isfinite(test_preds), test_preds, global_mean_fare)

# Build submission using sample_submission keys to ensure correct order
submission_template = pd.read_csv(sample_sub_path)
pred_df = pd.DataFrame({"key": test_full["key"], "fare_amount": test_preds})

submission = submission_template[["key"]].merge(pred_df, on="key", how="left")
submission["fare_amount"].fillna(global_mean_fare, inplace=True)

submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
submission.to_csv(submission_path, index=False)
print(f"Saved submission to {submission_path}")

submission_path_working = os.path.join(WORKING_DIR, "submission.csv")
submission.to_csv(submission_path_working, index=False)
print(f"Saved submission copy to {submission_path_working}")

# Print final CV metric explicitly at the end
print(f"Final 5-fold CV RMSE: {cv_rmse:.5f}")
