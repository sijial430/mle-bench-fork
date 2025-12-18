import os
import sys
import math
import glob
import numpy as np
import pandas as pd


# Utility: ECEF -> lat/lon/alt (WGS84)
def ecef_to_geodetic(x, y, z):
    # WGS84 constants
    a = 6378137.0
    e = 8.1819190842622e-2  # eccentricity
    asq = a * a
    esq = e * e

    b = math.sqrt(asq * (1 - esq))
    ep = math.sqrt((asq - b * b) / (b * b))
    p = math.sqrt(x * x + y * y)
    th = math.atan2(a * z, b * p)
    lon = math.atan2(y, x)
    lat = math.atan2(
        z + ep * ep * b * math.sin(th) ** 3, p - esq * a * math.cos(th) ** 3
    )
    N = a / math.sqrt(1 - esq * math.sin(lat) * math.sin(lat))
    alt = p / math.cos(lat) - N

    lat_deg = math.degrees(lat)
    lon_deg = math.degrees(lon)
    return lat_deg, lon_deg, alt


# Haversine distance (meters)
def haversine_meters(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))


# Find nearest indices for query times in a sorted time array
def nearest_indices(sorted_times, query_times):
    # sorted_times: 1D numpy sorted ascending
    # query_times: 1D numpy
    idx = np.searchsorted(sorted_times, query_times, side="left")
    idx0 = np.clip(idx - 1, 0, len(sorted_times) - 1)
    idx1 = np.clip(idx, 0, len(sorted_times) - 1)
    diff0 = np.abs(sorted_times[idx0] - query_times)
    diff1 = np.abs(sorted_times[idx1] - query_times)
    choose = diff1 < diff0
    return np.where(choose, idx1, idx0)


# Parse phone string into drive folder and phone name (split at last underscore)
def parse_phone_id(phone_id):
    # phone_id examples: "2020-05-15-US-MTV-1_Pixel4"
    if isinstance(phone_id, str) and "_" in phone_id:
        drive, phone = phone_id.rsplit("_", 1)
    else:
        parts = str(phone_id).split("/")
        drive = parts[0]
        phone = parts[-1]
    return drive, phone


INPUT_DIR = "./input"
SAMPLE_SUB_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
if not os.path.exists(SAMPLE_SUB_PATH):
    print("sample_submission.csv not found in ./input - aborting")
    sys.exit(1)

# Read and normalize sample submission columns to robustly find the phone and UnixTimeMillis columns
sample_sub = pd.read_csv(SAMPLE_SUB_PATH, dtype=str)
# Normalize columns map
orig_cols = list(sample_sub.columns)
lower_map = {c.lower().strip(): c for c in orig_cols}

# Find phone column (match 'phone')
phone_col = None
time_col = None
for key in lower_map:
    if key == "phone" or "phone" == key:
        phone_col = lower_map[key]
    if key in ("unixtimemillis", "unixtime", "timestamp", "unix_time_millis"):
        time_col = lower_map[key]
# fallback: pick columns containing substrings
if phone_col is None:
    for key, orig in lower_map.items():
        if "phone" in key:
            phone_col = orig
            break
if time_col is None:
    for key, orig in lower_map.items():
        if "unix" in key and "time" in key:
            time_col = orig
            break
# If still not found, try several common names
if phone_col is None:
    candidates = ["phone", "id", "device"]
    for cand in candidates:
        for key, orig in lower_map.items():
            if cand in key:
                phone_col = orig
                break
        if phone_col:
            break
if time_col is None:
    candidates = ["unixtimemillis", "time", "timestamp"]
    for cand in candidates:
        for key, orig in lower_map.items():
            if cand in key:
                time_col = orig
                break
        if time_col:
            break

if phone_col is None or time_col is None:
    print(
        "Could not find phone or UnixTimeMillis column in sample_submission. Columns:",
        orig_cols,
    )
    sys.exit(1)

# Ensure correct dtypes for time
sample_sub[time_col] = sample_sub[time_col].astype(np.int64)
# Rename for convenience
sample_sub = sample_sub.rename(columns={phone_col: "phone", time_col: "UnixTimeMillis"})

# Cache device GNSS data per (base_dir, drive, phone)
device_cache = {}


def load_device_gnss_for(drive, phone, base_dir="test"):
    key = (base_dir, drive, phone)
    if key in device_cache:
        return device_cache[key]
    path = os.path.join(INPUT_DIR, base_dir, drive, phone, "device_gnss.csv")
    if not os.path.exists(path):
        device_cache[key] = None
        return None
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        device_cache[key] = None
        return None
    if "utcTimeMillis" not in df.columns:
        # Try a few alternative column names
        candidate = None
        for c in df.columns:
            if c.lower().strip() in (
                "utc",
                "utcTimeMillis".lower(),
                "utctime",
                "utc_millis",
            ):
                candidate = c
                break
        if candidate is None:
            device_cache[key] = None
            return None
        else:
            df = df.rename(columns={candidate: "utcTimeMillis"})
    # Keep only rows with valid utcTimeMillis
    df = df.dropna(subset=["utcTimeMillis"])
    if len(df) == 0:
        device_cache[key] = {
            "times": np.array([], dtype=np.int64),
            "lat": np.array([]),
            "lon": np.array([]),
        }
        return device_cache[key]
    # Detect available position columns
    has_latlon = ("WlsLatitudeDegrees" in df.columns) and (
        "WlsLongitudeDegrees" in df.columns
    )
    has_ecef = (
        ("WlsPositionXEcefMeters" in df.columns)
        and ("WlsPositionYEcefMeters" in df.columns)
        and ("WlsPositionZEcefMeters" in df.columns)
    )
    # prepare times sorted
    times = df["utcTimeMillis"].astype(np.int64).to_numpy()
    order = np.argsort(times)
    times = times[order]
    if has_latlon:
        lat = df["WlsLatitudeDegrees"].to_numpy()[order].astype(float)
        lon = df["WlsLongitudeDegrees"].to_numpy()[order].astype(float)
        device_cache[key] = {"times": times, "lat": lat, "lon": lon}
    elif has_ecef:
        x = df["WlsPositionXEcefMeters"].to_numpy()[order].astype(float)
        y = df["WlsPositionYEcefMeters"].to_numpy()[order].astype(float)
        z = df["WlsPositionZEcefMeters"].to_numpy()[order].astype(float)
        lat = np.empty(len(x), dtype=float)
        lon = np.empty(len(x), dtype=float)
        for i, (xi, yi, zi) in enumerate(zip(x, y, z)):
            try:
                lati, loni, _ = ecef_to_geodetic(float(xi), float(yi), float(zi))
            except Exception:
                lati, loni = np.nan, np.nan
            lat[i] = lati
            lon[i] = loni
        device_cache[key] = {"times": times, "lat": lat, "lon": lon}
    else:
        # no usable positions, but still store times
        device_cache[key] = {
            "times": times,
            "lat": np.full_like(times, np.nan, dtype=float),
            "lon": np.full_like(times, np.nan, dtype=float),
        }
    return device_cache[key]


# Build predictions for sample_submission robustly by grouping rows per phone to reduce repeated lookups
out_rows = []
# We'll create a per-phone predictions by fetching device file once for each unique phone id in submission
unique_phones = sample_sub["phone"].unique()
# For phones missing device data, we could fallback to train median per that phone (not implemented extensively here)
for phone_id in unique_phones:
    # find rows in sample_sub for this phone
    mask = sample_sub["phone"] == phone_id
    times_query = sample_sub.loc[mask, "UnixTimeMillis"].to_numpy(dtype=np.int64)
    drive, phone = parse_phone_id(phone_id)
    data = load_device_gnss_for(drive, phone, base_dir="test")
    preds_lat = np.zeros(len(times_query), dtype=float)
    preds_lon = np.zeros(len(times_query), dtype=float)
    if data is None:
        # fallback zeros
        preds_lat[:] = 0.0
        preds_lon[:] = 0.0
    else:
        if len(data["times"]) == 0:
            preds_lat[:] = 0.0
            preds_lon[:] = 0.0
        else:
            idxs = nearest_indices(data["times"], times_query)
            preds_lat = data["lat"][idxs].astype(float)
            preds_lon = data["lon"][idxs].astype(float)
            # If nan, replace with median for that device if possible, otherwise 0
            if np.any(np.isnan(preds_lat)) or np.any(np.isnan(preds_lon)):
                lat_med = np.nanmedian(data["lat"])
                lon_med = np.nanmedian(data["lon"])
                # if median is nan, fallback to zeros
                if math.isnan(lat_med) or math.isnan(lon_med):
                    lat_med = 0.0
                    lon_med = 0.0
                nan_mask_lat = np.isnan(preds_lat)
                nan_mask_lon = np.isnan(preds_lon)
                preds_lat[nan_mask_lat] = lat_med
                preds_lon[nan_mask_lon] = lon_med
    # Append to out_rows preserving original ordering
    idxs_in_sub = sample_sub.index[mask].tolist()
    for idx_pos, sub_idx in enumerate(idxs_in_sub):
        out_rows.append(
            (
                sample_sub.at[sub_idx, "phone"],
                int(sample_sub.at[sub_idx, "UnixTimeMillis"]),
                float(preds_lat[idx_pos]),
                float(preds_lon[idx_pos]),
            )
        )

submission_df = pd.DataFrame(
    out_rows, columns=["phone", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
)
os.makedirs("./submission", exist_ok=True)
submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Saved test predictions to {submission_path}")

# Validation: evaluate on a small hold-out set from train drives.
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
if not os.path.exists(TRAIN_DIR):
    print("No train directory found, skipping validation metric.")
    sys.exit(0)

# Collect train phones (drive, phone) list
pairs = []
for drive in sorted(os.listdir(TRAIN_DIR)):
    drive_path = os.path.join(TRAIN_DIR, drive)
    if not os.path.isdir(drive_path):
        continue
    for phone_name in sorted(os.listdir(drive_path)):
        phone_path = os.path.join(drive_path, phone_name)
        if not os.path.isdir(phone_path):
            continue
        if os.path.exists(
            os.path.join(phone_path, "device_gnss.csv")
        ) and os.path.exists(os.path.join(phone_path, "ground_truth.csv")):
            pairs.append((drive, phone_name))
# Limit number for speed but ensure some evaluation
MAX_VAL_PHONES = 8
pairs = pairs[:MAX_VAL_PHONES]

per_phone_scores = []
for drive, phone in pairs:
    dev = load_device_gnss_for(drive, phone, base_dir="train")
    if dev is None or len(dev["times"]) == 0:
        continue
    gt_path = os.path.join(INPUT_DIR, "train", drive, phone, "ground_truth.csv")
    try:
        gt = pd.read_csv(gt_path)
    except Exception as e:
        continue
    if (
        ("UnixTimeMillis" not in gt.columns)
        or ("LatitudeDegrees" not in gt.columns)
        or ("LongitudeDegrees" not in gt.columns)
    ):
        # try normalizing columns
        cols_map = {c.lower().strip(): c for c in gt.columns}
        if (
            "unixtimemillis" in cols_map
            and "latitudedegrees" in cols_map
            and "longitudedegrees" in cols_map
        ):
            gt = gt.rename(
                columns={
                    cols_map["unixtimemillis"]: "UnixTimeMillis",
                    cols_map["latitudedegrees"]: "LatitudeDegrees",
                    cols_map["longitudedegrees"]: "LongitudeDegrees",
                }
            )
        else:
            continue
    query_times = gt["UnixTimeMillis"].astype(np.int64).to_numpy()
    if len(query_times) == 0:
        continue
    idxs = nearest_indices(dev["times"], query_times)
    preds_lat = dev["lat"][idxs]
    preds_lon = dev["lon"][idxs]
    distances = []
    for i in range(len(query_times)):
        plat = preds_lat[i]
        plon = preds_lon[i]
        glat = float(gt["LatitudeDegrees"].iloc[i])
        glon = float(gt["LongitudeDegrees"].iloc[i])
        if math.isnan(plat) or math.isnan(plon):
            distances.append(1000.0)
        else:
            distances.append(haversine_meters(glat, glon, float(plat), float(plon)))
    distances = np.array(distances)
    p50 = float(np.nanpercentile(distances, 50))
    p95 = float(np.nanpercentile(distances, 95))
    per_phone_scores.append((drive + "_" + phone, p50, p95))
    print(f"Phone {drive}_{phone}: p50={p50:.3f} m, p95={p95:.3f} m")

if len(per_phone_scores) == 0:
    print("No validation phones evaluated.")
else:
    avg_per_phone = [(p, (p50 + p95) / 2.0) for (p, p50, p95) in per_phone_scores]
    mean_metric = float(np.mean([v for (p, v) in avg_per_phone]))
    print(
        "\nValidation metric (mean across phones of (p50+p95)/2): {:.3f} meters".format(
            mean_metric
        )
    )
