import numpy as np
import pandas as pd
import duckdb
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import gc

# -- GATEKEEPER MODEL --
# OK, since throwing majority of the engineered features at the model did not
# enable it to become more accurate, try to pivot to a new strategy
# First check if the model is able to even predict there will be a change in speed
# to begin with, before proceeding

FEAT = r"C:/Users/Admin/Desktop/Sugarcane/UNI/Y2 Sem 1/FYP/Traffic Forecast Project/ml/build_features_final/master_features_v3.parquet"
con = duckdb.connect()
INC_NEAR = r"C:/Users/Admin/Desktop/Sugarcane/UNI/Y2 Sem 1/FYP/Traffic Forecast Project/ML/incident_nearby_buckets_200m.parquet"
ROAD_LINKS = r"C:/Users/Admin/Desktop/Sugarcane/UNI/Y2 Sem 1/FYP/Traffic Forecast Project/ml/road_links.parquet"

min_ts, max_ts = con.execute(f"""
    SELECT MIN(snapshot_ts), MAX(snapshot_ts)
    FROM read_parquet('{FEAT}')
""").fetchone()

split_ts = min_ts + int(0.8 * (max_ts - min_ts))
TRAIN_SAMPLE_FRAC = 0.1

train_df = con.execute(f"""
    SELECT
        f.*,
        r.start_lat, r.end_lat, r.start_lon, r.end_lon,
        COALESCE(i.incident_nearby, 0) AS incident_nearby,
        COALESCE(i.nearby_accident, 0) AS nearby_accident,
        COALESCE(i.nearby_roadwork, 0) AS nearby_roadwork,
        COALESCE(i.nearby_breakdown, 0) AS nearby_breakdown,
        COALESCE(i.mins_since_nearby_start, -1) AS mins_since_nearby_start
    FROM read_parquet('{FEAT}') f
    LEFT JOIN read_parquet('{ROAD_LINKS}') r
        ON f.link_id = r.link_id
    LEFT JOIN read_parquet('{INC_NEAR}') i
      ON f.link_id = i.link_id AND f.snapshot_ts = i.snapshot_ts
    WHERE f.snapshot_ts < {split_ts}
      AND (COALESCE(i.incident_nearby, 0) = 1 OR random() < {TRAIN_SAMPLE_FRAC})
""").df()

test_df = con.execute(f"""
    SELECT
        f.*,
        r.start_lat, r.end_lat, r.start_lon, r.end_lon,
        COALESCE(i.incident_nearby, 0) AS incident_nearby,
        COALESCE(i.nearby_accident, 0) AS nearby_accident,
        COALESCE(i.nearby_roadwork, 0) AS nearby_roadwork,
        COALESCE(i.nearby_breakdown, 0) AS nearby_breakdown,
        COALESCE(i.mins_since_nearby_start, -1) AS mins_since_nearby_start
    FROM read_parquet('{FEAT}') f
    LEFT JOIN read_parquet('{ROAD_LINKS}') r
        ON f.link_id = r.link_id
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
    # Calculate rate of change
    df["acceleration"] = df["delta_0_5"] - df["delta_5_10"]

    # Distance Proxy
    df["link_dist_proxy"] = np.sqrt(
        (df["start_lat"] - df["end_lat"])**2 + 
        (df["start_lon"] - df["end_lon"])**2
    )


    # Add the time context
    df["is_weekend"] = df["dow_sg"].isin([0, 6]).astype(int)
    df["is_peak"] = df["hour_sg"].isin([7, 8, 9, 17, 18, 19]).astype(int)

# Define features
features = [
    "sb","sb_tm5","sb_tm10","sb_tm15",
    "delta_0_5","delta_5_10","delta_10_15",
    "mid_lat", "mid_lon",
    # "hour_sg", "dow_sg",
    "acceleration", "link_dist_proxy",
    "rain_mm", "is_raining",
    "road_category","is_weekend","is_peak",
    "incident_nearby","mins_since_nearby_start",
    "nearby_accident","nearby_roadwork","nearby_breakdown"
]
X_train = train_df[features]
X_test  = test_df[features]

print(train_df.columns.tolist())
missing = [c for c in features if c not in train_df.columns]
print("Missing features:", missing)


y_train = (train_df["y_tp15"].astype(int) != train_df["sb"].astype(int)).astype(np.int32)
y_test  = (test_df["y_tp15"].astype(int) != test_df["sb"].astype(int)).astype(np.int32)

print("Train incident_nearby rate:", train_df["incident_nearby"].mean(), "rows:", (train_df["incident_nearby"]==1).sum())
print("Test incident_nearby rate:", test_df["incident_nearby"].mean(), "rows:", (test_df["incident_nearby"]==1).sum())

# ADJUST THE WEIGHTS, multiply incidents to make it more impactful during training
w = np.ones(len(train_df), dtype=np.float32)
w[train_df["incident_nearby"].to_numpy() == 1] *= 1.0  


# Convert to numpy 
X_train_np = X_train.to_numpy(dtype=np.float32, copy=False)
X_test_np  = X_test.to_numpy(dtype=np.float32, copy=False)
y_train_np = y_train.to_numpy(dtype=np.int32, copy=False)
y_test_np  = y_test.to_numpy(dtype=np.int32, copy=False)

del X_test
gc.collect()



if w is None:
    dtrain = xgb.QuantileDMatrix(X_train_np, label=y_train_np)
else:
    dtrain = xgb.QuantileDMatrix(X_train_np, label=y_train_np, weight=w.astype(np.float32, copy=False))



dtest = xgb.QuantileDMatrix(X_test_np, label=y_test_np)

params = {
    "objective": "binary:logistic",
    "learning_rate": 0.08,
    "max_depth": 8,
    "subsample": 0.84,
    "colsample_bytree": 0.64,
    "min_child_weight": 6,
    "tree_method": "hist",
    "device": "cuda",
    "eval_metric": "logloss",
}

bst = xgb.train(params, dtrain, num_boost_round=465)

del train_df
del X_train
del X_train_np
del y_train
del y_train_np
gc.collect()

pred_prob = bst.predict(dtest)
pred = (pred_prob >= 0.25).astype(np.int32)

# ACCURACY SCORES
# Metrics
y_true = y_test_np
y_pred = pred

print("\n--- Do a threshold sweep to get best threshold ---")
for t in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    t_pred = (pred_prob >= t).astype(int)
    r = recall_score(y_true, t_pred)
    p = precision_score(y_true, t_pred, zero_division=0)
    f = f1_score(y_true, t_pred, zero_division=0)
    print(f"Threshold {t:.2f}: Recall={r:.4f}, Precision={p:.4f}, F1={f:.4f}")

# Naive baseline: always predict no change
base_pred = np.zeros_like(y_test_np, dtype=np.int32)

# Overall metrics
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

base_acc = accuracy_score(y_true, base_pred)
base_prec = precision_score(y_true, base_pred, zero_division=0)
base_rec = recall_score(y_true, base_pred, zero_division=0)
base_f1 = f1_score(y_true, base_pred, zero_division=0)

# Masks
inc_mask = (test_df["incident_nearby"].to_numpy(dtype=np.int8, copy=False) == 1)
change_mask = (y_true == 1)

print("\nOverall Results:")
print(f"Model    acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}")
print(f"Baseline acc={base_acc:.4f} prec={base_prec:.4f} rec={base_rec:.4f} f1={base_f1:.4f}")

print("Change rate overall:", y_true.mean())
print("True change rows:", change_mask.sum())

# Persistence Baseline: If current speed != 5 mins ago, predict "Change"
persistence_pred = (test_df["sb"] != test_df["sb_tm5"]).astype(int)

persist_f1 = f1_score(y_true, persistence_pred)
print(f"Persistence Baseline F1: {persist_f1:.4f}")

if inc_mask.any():
    inc_acc = accuracy_score(y_true[inc_mask], y_pred[inc_mask])
    inc_prec = precision_score(y_true[inc_mask], y_pred[inc_mask], zero_division=0)
    inc_rec = recall_score(y_true[inc_mask], y_pred[inc_mask], zero_division=0)
    inc_f1 = f1_score(y_true[inc_mask], y_pred[inc_mask], zero_division=0)

    base_inc_acc = accuracy_score(y_true[inc_mask], base_pred[inc_mask])
    base_inc_prec = precision_score(y_true[inc_mask], base_pred[inc_mask], zero_division=0)
    base_inc_rec = recall_score(y_true[inc_mask], base_pred[inc_mask], zero_division=0)
    base_inc_f1 = f1_score(y_true[inc_mask], base_pred[inc_mask], zero_division=0)

    print("\nIncidentNearby Results:")
    print(f"Model    acc={inc_acc:.4f} prec={inc_prec:.4f} rec={inc_rec:.4f} f1={inc_f1:.4f}")
    print(f"Baseline acc={base_inc_acc:.4f} prec={base_inc_prec:.4f} rec={base_inc_rec:.4f} f1={base_inc_f1:.4f}")

if change_mask.any():
    print("Hit rate on true change rows:", (y_pred[change_mask] == y_true[change_mask]).mean())



# Save this features to a new parquet
OUT_PARQUET = r"C:/Users/Admin/Desktop/Sugarcane/UNI/Y2 Sem 1/FYP/Traffic Forecast Project/ML/gatekeeper_test_predictions.parquet"
test_df["true_change"] = y_test_np.astype(np.int8)
test_df["gk_prob"] = pred_prob.astype(np.float32)

# Append all probabilities first for testing
for t in [0.25, 0.30, 0.35, 0.40]:
    col_name = f"gk_pred_{int(t*100)}"
    test_df[col_name] = (pred_prob >= t).astype(np.int8)

test_df["is_fp"] = ((test_df["gk_pred_25"] == 1) & (test_df["true_change"] == 0)).astype(np.int8)
test_df["is_tp"] = ((test_df["gk_pred_25"] == 1) & (test_df["true_change"] == 1)).astype(np.int8)

# Save the direction first
# test_df["true_down"] = (test_df["y_tp15"] < test_df["sb"]).astype(np.int8)
# test_df["true_up"]   = (test_df["y_tp15"] > test_df["sb"]).astype(np.int8)


# cols_to_save = [
#     "link_id", "snapshot_ts", "sb", "y_tp15",
#     "true_change", "gk_prob", "is_fp", "is_tp",
#     "true_down", "true_up"
# ] + [f"gk_pred_{int(t*100)}" for t in [0.25, 0.30, 0.35, 0.40]]

# test_df[cols_to_save].to_parquet(OUT_PARQUET, index=False)
# print("Saved to Parquet")


bst.save_model("gatekeeper.json")


bst.feature_names = features
xgb.plot_importance(bst, max_num_features=15)
plt.show()
