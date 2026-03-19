import numpy as np
import pandas as pd
import duckdb
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import gc

# -- ASCENT SPECIALIST MODEL WITH SIMULATED DATA FROM ROUTER --
# OK, since throwing majority of the engineered features at the model did not
# enable it to become more accurate, try to pivot to a new strategy

FEAT = "build_features_final/master_features_v3.parquet"
con = duckdb.connect()
INC_NEAR = "incident_nearby_buckets_200m.parquet"
ROAD_LINKS = "road_links.parquet"
ROUTER_PARQUET = "router_predictions.parquet"


min_ts, max_ts = con.execute(f"""
    SELECT MIN(snapshot_ts), MAX(snapshot_ts)
    FROM read_parquet('{FEAT}')
""").fetchone()

split_ts = min_ts + int(0.8 * (max_ts - min_ts))
TRAIN_SAMPLE_FRAC = 0.1

changes_df = con.execute(f"""
    SELECT f.*, r.start_lat, r.end_lat, r.start_lon, r.end_lon,
        COALESCE(i.incident_nearby, 0) AS incident_nearby,
        COALESCE(i.nearby_accident, 0) AS nearby_accident,
        COALESCE(i.nearby_roadwork, 0) AS nearby_roadwork,
        COALESCE(i.nearby_breakdown, 0) AS nearby_breakdown,
        COALESCE(i.mins_since_nearby_start, -1) AS mins_since_nearby_start
    FROM read_parquet('{FEAT}') f
    LEFT JOIN read_parquet('{ROAD_LINKS}') r ON f.link_id = r.link_id
    LEFT JOIN read_parquet('{INC_NEAR}') i ON f.link_id = i.link_id AND f.snapshot_ts = i.snapshot_ts
    WHERE f.snapshot_ts < {split_ts} 
        AND f.y_tp15 > f.sb
    USING SAMPLE 500000
""").df()

# No change rows
stable_df = con.execute(f"""
    SELECT f.*, r.start_lat, r.end_lat, r.start_lon, r.end_lon,
        COALESCE(i.incident_nearby, 0) AS incident_nearby,
        COALESCE(i.nearby_accident, 0) AS nearby_accident,
        COALESCE(i.nearby_roadwork, 0) AS nearby_roadwork,
        COALESCE(i.nearby_breakdown, 0) AS nearby_breakdown,
        COALESCE(i.mins_since_nearby_start, -1) AS mins_since_nearby_start
    FROM read_parquet('{FEAT}') f
    LEFT JOIN read_parquet('{ROAD_LINKS}') r ON f.link_id = r.link_id
    LEFT JOIN read_parquet('{INC_NEAR}') i ON f.link_id = i.link_id AND f.snapshot_ts = i.snapshot_ts
    WHERE f.snapshot_ts < {split_ts}
        AND f.y_tp15 = f.sb
    USING SAMPLE 500000
""").df()

train_df = pd.concat([changes_df, stable_df])

# train_df = con.execute(f"""
#     SELECT
#         f.*,
#         r.start_lat, r.end_lat, r.start_lon, r.end_lon,
#         COALESCE(i.incident_nearby, 0) AS incident_nearby,
#         COALESCE(i.nearby_accident, 0) AS nearby_accident,
#         COALESCE(i.nearby_roadwork, 0) AS nearby_roadwork,
#         COALESCE(i.nearby_breakdown, 0) AS nearby_breakdown,
#         COALESCE(i.mins_since_nearby_start, -1) AS mins_since_nearby_start
#     FROM read_parquet('{FEAT}') f
#     LEFT JOIN read_parquet('{ROAD_LINKS}') r
#         ON f.link_id = r.link_id
#     LEFT JOIN read_parquet('{INC_NEAR}') i
#       ON f.link_id = i.link_id AND f.snapshot_ts = i.snapshot_ts
#     WHERE f.snapshot_ts < {split_ts}
#       AND random() < {TRAIN_SAMPLE_FRAC}
# """).df()

test_df = con.execute(f"""
    SELECT
        g.*,
        f.*,
        r.start_lat, r.end_lat, r.start_lon, r.end_lon
    FROM read_parquet('{ROUTER_PARQUET}') g
    LEFT JOIN read_parquet('{ROAD_LINKS}') r
        ON g.link_id = r.link_id
    INNER JOIN read_parquet('{FEAT}') f 
        ON g.link_id = f.link_id AND g.snapshot_ts = f.snapshot_ts
    WHERE gk_pred_25 = 1
        AND router_pred = 0
""").df()

# test_df = con.execute(f"""
#     SELECT
#         f.*,
#         r.start_lat, r.end_lat, r.start_lon, r.end_lon,
#         COALESCE(i.incident_nearby, 0) AS incident_nearby,
#         COALESCE(i.nearby_accident, 0) AS nearby_accident,
#         COALESCE(i.nearby_roadwork, 0) AS nearby_roadwork,
#         COALESCE(i.nearby_breakdown, 0) AS nearby_breakdown,
#         COALESCE(i.mins_since_nearby_start, -1) AS mins_since_nearby_start,
#         g.true_change,
#         g.true_down,
#         g.true_up,
#         g.gk_prob,
#         g.gk_pred_25,
#         g.gk_pred_30,
#         g.gk_pred_35,
#         g.gk_pred_40,
#         g.is_fp,
#         g.is_tp,
#         g.router_prob,
#         g.router_pred,
#         g.router_correct
#     FROM read_parquet('{FEAT}') f
#     INNER JOIN read_parquet('{ROUTER_PARQUET}') g
#       ON f.link_id = g.link_id AND f.snapshot_ts = g.snapshot_ts
#     LEFT JOIN read_parquet('{ROAD_LINKS}') r
#       ON f.link_id = r.link_id
#     LEFT JOIN read_parquet('{INC_NEAR}') i
#       ON f.link_id = i.link_id AND f.snapshot_ts = i.snapshot_ts
#     WHERE g.gk_pred_25 = 1
# """).df()
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
    # "gk_prob", "router_prob",
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


# LABELS 
# Change the logic, now we want to predict direction of change
# 1 - traffic is slowing
# 2 - traffic is recovering
y_train = np.maximum(train_df["y_tp15"] - train_df["sb"], 0).astype(np.float32)
y_test  = np.maximum(test_df["y_tp15"] - test_df["sb"], 0).astype(np.float32)

print("Train incident_nearby rate:", train_df["incident_nearby"].mean(), "rows:", (train_df["incident_nearby"]==1).sum())
print("Test incident_nearby rate:", test_df["incident_nearby"].mean(), "rows:", (test_df["incident_nearby"]==1).sum())

# ADJUST THE WEIGHTS, multiply incidents to make it more impactful during training
# turns out, not needed anymore. results stay nearly the same from 1.0 to 3.0
w = np.ones(len(train_df), dtype=np.float32)
w[train_df["incident_nearby"].to_numpy() == 1] *= 1.0  


# Convert to numpy 
X_train_np = X_train.to_numpy(dtype=np.float32, copy=False)
X_test_np  = X_test.to_numpy(dtype=np.float32, copy=False)
y_train_np = y_train.to_numpy(dtype=np.int32, copy=False)
y_test_np  = y_test.to_numpy(dtype=np.int32, copy=False)

if w is None:
    dtrain = xgb.QuantileDMatrix(X_train_np, label=y_train_np)
else:
    dtrain = xgb.QuantileDMatrix(X_train_np, label=y_train_np, weight=w.astype(np.float32, copy=False))

dtest = xgb.QuantileDMatrix(X_test_np, label=y_test_np)

params = {
    "objective": "reg:squarederror",
    "learning_rate": 0.05,
    "max_depth": 8,
    "subsample": 0.84,
    "colsample_bytree": 0.64,
    "min_child_weight": 6,
    "tree_method": "hist",
    "device": "cuda",
    "eval_metric": "mae",
}

bst = xgb.train(params, dtrain, num_boost_round=465)

# Get back some memory
del train_df
del X_train_np
del dtrain
gc.collect()


# Predict
pred_mag = bst.predict(dtest)
pred_mag = np.clip(pred_mag, 0, None)
rounded_pred = np.rint(pred_mag).astype(np.int32)




# Predict
pred_mag = bst.predict(dtest)
pred_mag = np.clip(pred_mag, 0, None)
rounded_pred = np.rint(pred_mag).astype(np.int32)

# Veto Logic
# Define "strict" threshold to filter out
strict_threshold = 0.75
strict_alerts = (pred_mag >= strict_threshold)

# Raw Counts
tp = np.sum((strict_alerts == True) & (y_test_np >= 1))
fp = np.sum((strict_alerts == True) & (y_test_np == 0))
total_alerts = tp + fp

# Precision/Recall
precision = tp / total_alerts if total_alerts > 0 else 0
recall = tp / np.sum(y_test_np >= 1)

print(f"\nResults are applying threshold (Threshold {strict_threshold})")
print(f"Total High-Confidence Alerts: {total_alerts:,}")
print(f"True Positives (Real Slowdowns Caught): {tp:,}")
print(f"False Positives (Trash/Noise destroyed): {fp:,}")
print(f"---")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# Compare to the 'Standard' 0.5 Rounding
standard_alerts = (pred_mag >= 0.5)
std_fp = np.sum((standard_alerts == True) & (y_test_np == 0))
print(f"\nSaved from {std_fp - fp:,} False Alarms by using the 0.8 Veto.")
# ACCURACY SCORES
# Metrics

mae = np.mean(np.abs(pred_mag - y_test_np))
rmse = np.sqrt(np.mean((pred_mag - y_test_np) ** 2))
exact_mag_acc = np.mean(rounded_pred == y_test_np)
within1_mag = np.mean(np.abs(rounded_pred - y_test_np) <= 1)

print("\nSpecialist Magnitude Results:")
print(f"MAE={mae:.4f}")
print(f"RMSE={rmse:.4f}")
print(f"Exact magnitude acc={exact_mag_acc:.4f}")
print(f"Within-1 magnitude acc={within1_mag:.4f}")

mae = np.mean(np.abs(pred_mag - y_test_np))
rmse = np.sqrt(np.mean((pred_mag - y_test_np) ** 2))
exact_mag_acc = np.mean(rounded_pred == y_test_np)
within1_mag = np.mean(np.abs(rounded_pred - y_test_np) <= 1)

print("\nSpecialist Magnitude Results:")
print(f"MAE={mae:.4f}")
print(f"RMSE={rmse:.4f}")
print(f"Exact magnitude acc={exact_mag_acc:.4f}")
print(f"Within-1 magnitude acc={within1_mag:.4f}")

print("\nRouter-passed set:")
print("Rows:", len(test_df))
print("True-change rate:", test_df["true_change"].mean())

print("\nPredicted rounded magnitude counts:")
print(pd.Series(rounded_pred).value_counts().sort_index())
print("Avg pred_mag on true no-change rows:", pred_mag[y_test_np == 0].mean())
print("Avg pred_mag on true changed rows:", pred_mag[y_test_np > 0].mean())
bins = pd.cut(pred_mag, bins=[-0.001, 0.75, 1.5, 3.0, 10.0])
print(pd.crosstab(bins, y_test_np > 0))

bst.feature_names = features
xgb.plot_importance(bst, max_num_features=15)
plt.show()

bst.save_model("asc_specialist.json")
print("Specialist model saved!")