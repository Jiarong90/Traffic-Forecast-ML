import numpy as np
import pandas as pd
import duckdb
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt
import gc

FEAT = "build_features_final/master_features_v3.parquet"
con = duckdb.connect()
INC_NEAR = r"C:/Users/Admin/Desktop/Sugarcane/UNI/Y2 Sem 1/FYP/Traffic Forecast Project/ML/incident_nearby_buckets_200m.parquet"
INC_NEAR = "incident_nearby_buckets_200m.parquet"
ROAD_LINKS = "road_links.parquet"
ROUTER_PARQUET = "router_predictions.parquet"

min_ts, max_ts = con.execute(f"""
    SELECT MIN(snapshot_ts), MAX(snapshot_ts)
    FROM read_parquet('{FEAT}')
""").fetchone()

split_ts = min_ts + int(0.8 * (max_ts - min_ts))
TRAIN_SAMPLE_FRAC = 0.1

# All Changed Rows (Ablation: Both Ascent and Descent)
changes_df = con.execute(f"""
    SELECT 
        f.*, 
        r.start_lat, r.end_lat, r.start_lon, r.end_lon,
        COALESCE(i.incident_nearby, 0) AS incident_nearby,
        COALESCE(i.nearby_accident, 0) AS nearby_accident,
        COALESCE(i.nearby_roadwork, 0) AS nearby_roadwork,
        COALESCE(i.nearby_breakdown, 0) AS nearby_breakdown,
        COALESCE(i.mins_since_nearby_start, -1) AS mins_since_nearby_start
    FROM read_parquet('{FEAT}') f
    LEFT JOIN read_parquet('{ROAD_LINKS}') r ON f.link_id = r.link_id
    LEFT JOIN read_parquet('{INC_NEAR}') i ON f.link_id = i.link_id AND f.snapshot_ts = i.snapshot_ts
    WHERE f.snapshot_ts < {split_ts} 
      AND f.y_tp15 != f.sb  -- Captures all real movements
    USING SAMPLE 500000
""").df()

# All Stable Rows (The 1:1 Counterpart)
stable_df = con.execute(f"""
    SELECT 
        f.*, 
        r.start_lat, r.end_lat, r.start_lon, r.end_lon,
        COALESCE(i.incident_nearby, 0) AS incident_nearby,
        COALESCE(i.nearby_accident, 0) AS nearby_accident,
        COALESCE(i.nearby_roadwork, 0) AS nearby_roadwork,
        COALESCE(i.nearby_breakdown, 0) AS nearby_breakdown,
        COALESCE(i.mins_since_nearby_start, -1) AS mins_since_nearby_start
    FROM read_parquet('{FEAT}') f
    LEFT JOIN read_parquet('{ROAD_LINKS}') r ON f.link_id = r.link_id
    LEFT JOIN read_parquet('{INC_NEAR}') i ON f.link_id = i.link_id AND f.snapshot_ts = i.snapshot_ts
    WHERE f.snapshot_ts < {split_ts}
      AND f.y_tp15 = f.sb  -- Captures the noise floor
    USING SAMPLE 500000
""").df()

train_df = pd.concat([changes_df, stable_df])

test_df = con.execute(f"""
    SELECT
        f.*,
        COALESCE(i.incident_nearby, 0) AS incident_nearby,
        COALESCE(i.nearby_accident, 0) AS nearby_accident,
        COALESCE(i.nearby_roadwork, 0) AS nearby_roadwork,
        COALESCE(i.nearby_breakdown, 0) AS nearby_breakdown,
        COALESCE(i.mins_since_nearby_start, -1) AS mins_since_nearby_start
    FROM read_parquet('{FEAT}') f
    LEFT JOIN read_parquet('{INC_NEAR}') i
      ON f.link_id = i.link_id AND f.snapshot_ts = i.snapshot_ts
    WHERE f.snapshot_ts >= {split_ts} 
""").df()

# Feature Engineering
for df in (train_df, test_df):
    df["road_category"] = df["road_category"].astype(int)
    # Fill any missing data just in case
    
    # Add a speed delta to see current momentum
    df["delta_0_5"] = df["sb"] - df["sb_tm5"]
    df["delta_5_10"] = df["sb_tm5"] - df["sb_tm10"]
    df["delta_10_15"] = df["sb_tm10"] - df["sb_tm15"]

    # Add the time context back
    df["is_weekend"] = df["dow_sg"].isin([0, 6]).astype(int)
    df["is_peak"] = df["hour_sg"].isin([7, 8, 9, 17, 18, 19]).astype(int)

# Define features (v3 - The Complete Package)
features = [
    "sb","sb_tm5","sb_tm10","sb_tm15",
    "delta_0_5","delta_5_10","delta_10_15",
    # "mid_lat", "mid_lon",
    "road_category","is_weekend","is_peak",
    "incident_nearby","mins_since_nearby_start",
    "nearby_accident","nearby_roadwork","nearby_breakdown"
]
X_train = train_df[features]
X_test  = test_df[features]

print(train_df.columns.tolist())
missing = [c for c in features if c not in train_df.columns]
print("Missing features:", missing)


y_train = train_df["y_tp15"].astype(int) - 1
y_test  = test_df["y_tp15"].astype(int) - 1

print("Train incident_nearby rate:", train_df["incident_nearby"].mean(), "rows:", (train_df["incident_nearby"]==1).sum())
print("Test incident_nearby rate:", test_df["incident_nearby"].mean(), "rows:", (test_df["incident_nearby"]==1).sum())

# ADJUST THE WEIGHTS, multiply incidents to make it more impactful during training
w = np.ones(len(train_df), dtype=np.float32)
w[train_df["incident_nearby"].to_numpy() == 1] *= 5.0  


num_class = 8
# Convert to numpy (float32 saves memory)
X_train_np = X_train.to_numpy(dtype=np.float32, copy=False)
X_test_np  = X_test.to_numpy(dtype=np.float32, copy=False)
y_train_np = y_train.to_numpy(dtype=np.int32, copy=False)
y_test_np  = y_test.to_numpy(dtype=np.int32, copy=False)

# Build QuantileDMatrix (GPU-friendly)
if w is None:
    dtrain = xgb.QuantileDMatrix(X_train_np, label=y_train_np)
else:
    dtrain = xgb.QuantileDMatrix(X_train_np, label=y_train_np, weight=w.astype(np.float32, copy=False))

# If X_test is huge, consider sampling/chunking. For now:
dtest = xgb.QuantileDMatrix(X_test_np, label=y_test_np)

params = {
    "objective": "multi:softmax",
    "num_class": num_class,
    "learning_rate": 0.08,
    "max_depth": 8,
    "subsample": 0.844237,
    "colsample_bytree": 0.638003,
    "min_child_weight": 6,
    "tree_method": "hist",
    "device": "cuda",
    "eval_metric": "mlogloss",
}

bst = xgb.train(params, dtrain, num_boost_round=465)

del train_df
del X_train_np
del dtrain
gc.collect()

pred = bst.predict(dtest).astype(np.int32)

# ACCURACY SCORES
# Metrics 
y_true = y_test_np                     
y_pred = pred                          

base_pred = test_df["sb"].to_numpy(dtype=np.int32, copy=False) - 1

acc = (y_pred == y_true).mean()
mae = np.mean(np.abs(y_pred - y_true))
within1 = np.mean(np.abs(y_pred - y_true) <= 1)

base_acc = (base_pred == y_true).mean()
base_mae = np.mean(np.abs(base_pred - y_true))
base_within1 = np.mean(np.abs(base_pred - y_true) <= 1)

chg = (base_pred != y_true)
acc_change = (y_pred[chg] == y_true[chg]).mean() if chg.any() else float("nan")


inc = (test_df["incident_nearby"].to_numpy(dtype=np.int8) == 1)
mask = chg & inc

print("Change+IncidentNearby rows:", mask.sum())
if mask.any():
    print("Model acc on Change+IncidentNearby:", (y_pred[mask] == y_true[mask]).mean())
    print("Persist acc on Change+IncidentNearby:", (base_pred[mask] == y_true[mask]).mean())


inc_mask = (test_df["incident_nearby"].to_numpy(dtype=np.int8, copy=False) == 1)
inc_acc = (y_pred[inc_mask] == y_true[inc_mask]).mean() if inc_mask.any() else float("nan")
base_inc_acc = (base_pred[inc_mask] == y_true[inc_mask]).mean() if inc_mask.any() else float("nan")

inc = inc_mask
print("IncidentNearby MAE model:", np.mean(np.abs(y_pred[inc] - y_true[inc])))
print("IncidentNearby MAE base :", np.mean(np.abs(base_pred[inc] - y_true[inc])))
print("IncidentNearby within1 model:", np.mean(np.abs(y_pred[inc] - y_true[inc]) <= 1))
print("IncidentNearby within1 base :", np.mean(np.abs(base_pred[inc] - y_true[inc]) <= 1))

print("\nResults:")
print(f"Overall acc={acc:.4f} within1={within1:.4f} MAE={mae:.4f}")
print(f"Persistence acc={base_acc:.4f} within1={base_within1:.4f} MAE={base_mae:.4f}")
print(f"Change-row acc: model={acc_change:.4f}")
print(f"IncidentNearby acc: model={inc_acc:.4f} | persistence={base_inc_acc:.4f} | rows={inc_mask.sum()}")

# change rate sanity check
print("Change rate overall:", chg.mean())
print("Change rate on incident_nearby rows:", chg[inc_mask].mean())

bst.feature_names = features
xgb.plot_importance(bst, max_num_features=15)
plt.show()