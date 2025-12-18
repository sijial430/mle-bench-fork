import os
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMRegressor

# Paths
INPUT_DIR = "./input"
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")
SUBMISSION_DIR = "./submission"
WORKING_DIR = "./working"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(WORKING_DIR, exist_ok=True)

# ----------------- coordinate utilities ----------------- #

EARTH_RADIUS = 6378137.0
ECCENTRICITY = 8.1819190842622e-2  # WGS-84 first eccentricity


def ecef_to_geodetic(x, y, z):
    """Convert ECEF (meters) to WGS84 lat, lon (degrees), h (meters)."""
    a = EARTH_RADIUS
    e = ECCENTRICITY
    b = math.sqrt(a * a * (1 - e * e))
    ep = math.sqrt((a * a - b * b) / (b * b))
    p = math.sqrt(x * x + y * y)
    th = math.atan2(a * z, b * p)
    lon = math.atan2(y, x)
    lat = math.atan2(
        z + ep * ep * b * math.sin(th) * math.sin(th) * math.sin(th),
        p - e * e * a * math.cos(th) * math.cos(th) * math.cos(th),
    )
    N = a / math.sqrt(1 - e * e * math.sin(lat) * math.sin(lat))
    h = p / math.cos(lat) - N
    lat_deg = math.degrees(lat)
    lon_deg = math.degrees(lon)
    return lat_deg, lon_deg, h


def ecef_to_geodetic_vec(x, y, z):
    lat = np.empty_like(x, dtype=float)
    lon = np.empty_like(x, dtype=float)
    h = np.empty_like(x, dtype=float)
    for i in range(len(x)):
        lat[i], lon[i], h[i] = ecef_to_geodetic(x[i], y[i], z[i])
    return lat, lon, h


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def comp_metric(df):
    metrics = []
    for phone, g in df.groupby("phone"):
        dists = haversine(
            g["lat_gt"].values,
            g["lon_gt"].values,
            g["lat_pred"].values,
            g["lon_pred"].values,
        )
        if len(dists) == 0:
            continue
        p50 = np.percentile(dists, 50)
        p95 = np.percentile(dists, 95)
        metrics.append((p50 + p95) / 2.0)
    if not metrics:
        return float("nan")
    return float(np.mean(metrics))


# ----------------- data loading helpers ----------------- #


def list_phone_dirs(root_dir):
    res = []
    if not os.path.isdir(root_dir):
        return res
    for drive in sorted(os.listdir(root_dir)):
        drive_path = os.path.join(root_dir, drive)
        if not os.path.isdir(drive_path):
            continue
        for phone in sorted(os.listdir(drive_path)):
            phone_path = os.path.join(drive_path, phone)
            if not os.path.isdir(phone_path):
                continue
            res.append((drive, phone, phone_path))
    return res


def load_train_phone(drive, phone, phone_path):
    gt_path = os.path.join(phone_path, "ground_truth.csv")
    gnss_path = os.path.join(phone_path, "device_gnss.csv")
    if not (os.path.exists(gt_path) and os.path.exists(gnss_path)):
        return None
    gt = pd.read_csv(gt_path)
    gnss = pd.read_csv(gnss_path)

    # Need WLS ECEF positions
    needed_cols = [
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
        "ArrivalTimeNanosSinceGpsEpoch",
    ]
    for c in needed_cols:
        if c not in gnss.columns:
            return None

    gt = gt.dropna(subset=["LatitudeDegrees", "LongitudeDegrees", "UnixTimeMillis"])
    gnss = gnss.dropna(subset=needed_cols)

    # Convert GPS epoch nanos to ms, estimate offset to UnixTimeMillis
    gnss["gps_ms"] = gnss["ArrivalTimeNanosSinceGpsEpoch"] / 1e6
    if "utcTimeMillis" in gnss.columns:
        gnss["offset"] = gnss["utcTimeMillis"] - gnss["gps_ms"]
        offset = gnss["offset"].median()
        gnss.drop(columns=["offset"], inplace=True)
    else:
        # approximate using overlap with GT
        offset = gt["UnixTimeMillis"].min() - gnss["gps_ms"].min()
    gnss["UnixTimeMillis"] = (gnss["gps_ms"] + offset).round().astype("int64")

    # Aggregate ECEF by epoch (mean, though usually single row per epoch)
    agg_cols = [
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
        "Cn0DbHz",
        "PseudorangeRateMetersPerSecond",
        "PseudorangeRateUncertaintyMetersPerSecond",
    ]
    present = [c for c in agg_cols if c in gnss.columns]
    agg_dict = {c: ["mean", "std"] for c in present}
    agg_dict.update({"Svid": "nunique"})
    g_agg = gnss.groupby("UnixTimeMillis").agg(agg_dict)
    g_agg.columns = ["{}_{}".format(c[0], c[1]) for c in g_agg.columns]
    g_agg = g_agg.reset_index()

    df = pd.merge(gt, g_agg, on="UnixTimeMillis", how="inner")
    if df.empty:
        return None

    # derive baseline lat/lon from mean ECEF
    if all(f"WlsPosition{ax}EcefMeters_mean" in df.columns for ax in ["X", "Y", "Z"]):
        x = df["WlsPositionXEcefMeters_mean"].values
        y = df["WlsPositionYEcefMeters_mean"].values
        z = df["WlsPositionZEcefMeters_mean"].values
        b_lat, b_lon, _ = ecef_to_geodetic_vec(x, y, z)
        df["base_lat"] = b_lat
        df["base_lon"] = b_lon
    else:
        df["base_lat"] = df["LatitudeDegrees"].mean()
        df["base_lon"] = df["LongitudeDegrees"].mean()

    df["phone"] = f"{drive}_{phone}"
    return df


def load_test_phone(drive, phone, phone_path):
    gnss_path = os.path.join(phone_path, "device_gnss.csv")
    if not os.path.exists(gnss_path):
        return None
    gnss = pd.read_csv(gnss_path)

    needed_cols = [
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
        "ArrivalTimeNanosSinceGpsEpoch",
    ]
    for c in needed_cols:
        if c not in gnss.columns:
            return None

    gnss = gnss.dropna(subset=needed_cols)
    gnss["gps_ms"] = gnss["ArrivalTimeNanosSinceGpsEpoch"] / 1e6
    if "utcTimeMillis" in gnss.columns:
        gnss["offset"] = gnss["utcTimeMillis"] - gnss["gps_ms"]
        offset = gnss["offset"].median()
        gnss.drop(columns=["offset"], inplace=True)
    else:
        offset = -gnss["gps_ms"].min()
    gnss["UnixTimeMillis"] = (gnss["gps_ms"] + offset).round().astype("int64")

    agg_cols = [
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
        "Cn0DbHz",
        "PseudorangeRateMetersPerSecond",
        "PseudorangeRateUncertaintyMetersPerSecond",
    ]
    present = [c for c in agg_cols if c in gnss.columns]
    agg_dict = {c: ["mean", "std"] for c in present}
    agg_dict.update({"Svid": "nunique"})
    g_agg = gnss.groupby("UnixTimeMillis").agg(agg_dict)
    g_agg.columns = ["{}_{}".format(c[0], c[1]) for c in g_agg.columns]
    g_agg = g_agg.reset_index()

    # baseline lat/lon
    if all(
        f"WlsPosition{ax}EcefMeters_mean" in g_agg.columns for ax in ["X", "Y", "Z"]
    ):
        x = g_agg["WlsPositionXEcefMeters_mean"].values
        y = g_agg["WlsPositionYEcefMeters_mean"].values
        z = g_agg["WlsPositionZEcefMeters_mean"].values
        b_lat, b_lon, _ = ecef_to_geodetic_vec(x, y, z)
        g_agg["base_lat"] = b_lat
        g_agg["base_lon"] = b_lon
    else:
        g_agg["base_lat"] = 0.0
        g_agg["base_lon"] = 0.0

    g_agg["phone"] = f"{drive}_{phone}"
    return g_agg


# ----------------- build full train data ----------------- #

train_records = []
for drive, phone, path in list_phone_dirs(TRAIN_DIR):
    try:
        rec = load_train_phone(drive, phone, path)
        if rec is not None:
            train_records.append(rec)
    except Exception as e:
        print(f"Error loading train {drive}/{phone}: {e}")
        continue

if not train_records:
    raise RuntimeError("No training data loaded")

train_df = pd.concat(train_records, ignore_index=True)
train_df = train_df.sort_values(["phone", "UnixTimeMillis"]).reset_index(drop=True)

# Targets and residuals vs baseline
train_df["tgt_lat"] = train_df["LatitudeDegrees"]
train_df["tgt_lon"] = train_df["LongitudeDegrees"]
train_df["dlat"] = train_df["tgt_lat"] - train_df["base_lat"]
train_df["dlon"] = train_df["tgt_lon"] - train_df["base_lon"]

# Build feature columns: ECEF stats + C/N0 and pseudorange stats and temporal diffs
exclude_cols = {
    "LatitudeDegrees",
    "LongitudeDegrees",
    "AltitudeMeters",
    "MessageType",
    "SpeedMps",
    "AccuracyMeters",
    "BearingDegrees",
    "tgt_lat",
    "tgt_lon",
    "dlat",
    "dlon",
    "phone",
    "UnixTimeMillis",
}
base_feature_candidates = [
    c
    for c in train_df.columns
    if c not in exclude_cols
    and (
        "WlsPosition" in c
        or c.startswith("Cn0DbHz_")
        or c.startswith("PseudorangeRateMetersPerSecond_")
        or c.startswith("PseudorangeRateUncertaintyMetersPerSecond_")
        or c in ["Svid_nunique"]
        or c in ["base_lat", "base_lon"]
    )
]

# Temporal diffs for key numeric features
for feat in base_feature_candidates:
    if train_df[feat].dtype != "O":
        train_df[feat + "_diff"] = train_df.groupby("phone")[feat].diff()

feature_cols = base_feature_candidates + [
    c for c in train_df.columns if c.endswith("_diff")
]

# Ensure feature_cols not empty
if not feature_cols:
    raise RuntimeError("No feature columns constructed for training")

# Fill NaNs
train_df[feature_cols] = train_df[feature_cols].fillna(train_df[feature_cols].median())

X = train_df[feature_cols].values
y_dlat = train_df["dlat"].values
y_dlon = train_df["dlon"].values
groups = train_df["phone"].values

# ----------------- cross-validation ----------------- #

gkf = GroupKFold(n_splits=5)
oof_lat = np.zeros(len(train_df))
oof_lon = np.zeros(len(train_df))

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y_dlat, groups)):
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_dlat_tr, y_dlat_val = y_dlat[tr_idx], y_dlat[val_idx]
    y_dlon_tr, y_dlon_val = y_dlon[tr_idx], y_dlon[val_idx]

    params = dict(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_lambda=0.1,
        random_state=42 + fold,
        n_jobs=-1,
    )
    model_lat = LGBMRegressor(**params)
    model_lon = LGBMRegressor(**params)

    model_lat.fit(X_tr, y_dlat_tr)
    model_lon.fit(X_tr, y_dlon_tr)

    oof_lat[val_idx] = train_df["base_lat"].values[val_idx] + model_lat.predict(X_val)
    oof_lon[val_idx] = train_df["base_lon"].values[val_idx] + model_lon.predict(X_val)

# Evaluate CV metric
oof_df = train_df[
    ["phone", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
].copy()
oof_df["lat_gt"] = oof_df["LatitudeDegrees"]
oof_df["lon_gt"] = oof_df["LongitudeDegrees"]
oof_df["lat_pred"] = oof_lat
oof_df["lon_pred"] = oof_lon
score = comp_metric(oof_df)
print("CV metric (mean of 50th and 95th percentile distance errors):", score)

# ----------------- train final models on full data ----------------- #

final_model_lat = LGBMRegressor(
    n_estimators=350,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=64,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_samples=20,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
)
final_model_lon = LGBMRegressor(
    n_estimators=350,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=64,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_samples=20,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
)

final_model_lat.fit(X, y_dlat)
final_model_lon.fit(X, y_dlon)

# ----------------- prepare test features aligned with sample submission ----------------- #

sample_sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")
if not os.path.exists(sample_sub_path):
    raise RuntimeError("sample_submission.csv not found in ./input")
sample_sub = pd.read_csv(sample_sub_path)

# Identifier column
if "phone" in sample_sub.columns:
    id_col = "phone"
elif "tripId" in sample_sub.columns:
    id_col = "tripId"
else:
    id_col = sample_sub.columns[0]

time_col = "UnixTimeMillis"
if time_col not in sample_sub.columns:
    raise RuntimeError("UnixTimeMillis column not found in sample_submission.csv")

# Unified phone column
sample_sub["phone"] = sample_sub[id_col].astype(str)

# Build test GNSS features
test_records = []
for drive, phone, path in list_phone_dirs(TEST_DIR):
    try:
        rec = load_test_phone(drive, phone, path)
        if rec is not None:
            test_records.append(rec)
    except Exception as e:
        print(f"Error loading test {drive}/{phone}: {e}")
        continue

if test_records:
    test_gnss = pd.concat(test_records, ignore_index=True)
else:
    test_gnss = pd.DataFrame(columns=["UnixTimeMillis", "phone"])

if "UnixTimeMillis" in test_gnss.columns:
    test_gnss["UnixTimeMillis"] = test_gnss["UnixTimeMillis"].astype("int64")

# Ensure all feature columns exist in test_gnss
for c in feature_cols:
    if c not in test_gnss.columns:
        test_gnss[c] = np.nan

# Sort and fill
if not test_gnss.empty:
    test_gnss = test_gnss.sort_values(["phone", "UnixTimeMillis"]).reset_index(
        drop=True
    )
    test_gnss[feature_cols] = test_gnss[feature_cols].fillna(
        train_df[feature_cols].median()
    )

# Merge with sample submission times
test_merge = pd.merge(
    sample_sub[["phone", time_col]],
    test_gnss[["phone", "UnixTimeMillis"] + feature_cols],
    left_on=["phone", time_col],
    right_on=["phone", "UnixTimeMillis"],
    how="left",
)

# sort and forward/backward fill within phone
test_merge = test_merge.sort_values(["phone", "UnixTimeMillis"])
test_merge[feature_cols] = test_merge.groupby("phone")[feature_cols].ffill()
test_merge[feature_cols] = test_merge.groupby("phone")[feature_cols].bfill()
test_merge[feature_cols] = test_merge[feature_cols].fillna(
    train_df[feature_cols].median()
)

X_test = test_merge[feature_cols].values

# Baseline lat/lon for test: from features if present, else global means
if "base_lat" in test_merge.columns and "base_lon" in test_merge.columns:
    base_lat_test = test_merge["base_lat"].values
    base_lon_test = test_merge["base_lon"].values
else:
    base_lat_test = np.full(len(test_merge), train_df["LatitudeDegrees"].mean())
    base_lon_test = np.full(len(test_merge), train_df["LongitudeDegrees"].mean())

pred_dlat = final_model_lat.predict(X_test)
pred_dlon = final_model_lon.predict(X_test)
pred_lat = base_lat_test + pred_dlat
pred_lon = base_lon_test + pred_dlon

# Build submission with original identifier column name
submission = pd.DataFrame()
submission[id_col] = sample_sub[id_col]
submission[time_col] = sample_sub[time_col]
submission["LatitudeDegrees"] = pred_lat
submission["LongitudeDegrees"] = pred_lon

sub_path = os.path.join(SUBMISSION_DIR, "submission.csv")
submission.to_csv(sub_path, index=False)
print("Saved submission to", sub_path)

working_sub_path = os.path.join(WORKING_DIR, "submission.csv")
submission.to_csv(working_sub_path, index=False)
print("Saved submission to", working_sub_path)
