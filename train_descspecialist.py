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
        AND f.y_tp15 < f.sb
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
        AND router_pred = 1
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
y_train = np.maximum(train_df["sb"] - train_df["y_tp15"], 0).astype(np.float32)
y_test  = np.maximum(test_df["sb"] - test_df["y_tp15"], 0).astype(np.float32)

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

# Predictions
pred_mag = bst.predict(dtest)
pred_mag = np.clip(pred_mag, 0, None)
rounded_pred = np.rint(pred_mag).astype(np.int32)

# Veto Logic
strict_threshold = 0.75
strict_alerts = (pred_mag >= strict_threshold)

tp = np.sum((strict_alerts == True) & (y_test_np >= 1))
fp = np.sum((strict_alerts == True) & (y_test_np == 0))
total_alerts = tp + fp

precision = tp / total_alerts if total_alerts > 0 else 0
recall = tp / np.sum(y_test_np >= 1)


print(f"Total High-Confidence Jam Alerts: {total_alerts:,}")
print(f"True Positives: {tp:,}")
print(f"False Positives: {fp:,}")
print(f"---")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# Bins
bin_edges = [-0.001, 0.75, 1.5, 3.0, 10.0]
bin_labels = ["Noise (Vetoed)", "Minor Jam (1-band)", "Significant Jam (2-3)", "Major Jam (4+)"]

bins = pd.cut(pred_mag, bins=bin_edges, labels=bin_labels)

print("\n--- PERFORMANCE BY JAM SEVERITY ---")
print(pd.crosstab(bins, y_test_np > 0, rownames=['Model Category'], colnames=['Real Jam?']))

# Metrics
mae = np.mean(np.abs(pred_mag - y_test_np))
exact_mag_acc = np.mean(rounded_pred == y_test_np)
print(f"\nAvg Error: {mae:.4f} bands")
print(f"Exact Hit Rate: {exact_mag_acc:.2%}")

# Save
# bst.save_model("descent_specialist.json")
# print("saved model/descent_specialist.json")

# # Visual importance check
# bst.feature_names = features
# xgb.plot_importance(bst, max_num_features=15)
# plt.show()


print("\n--- RECALL BY TRUE JAM SEVERITY ---")

severity_buckets = [
    ("Minor Jam (1-band)", y_test_np == 1),
    ("Significant Jam (2-3)", (y_test_np >= 2) & (y_test_np <= 3)),
    ("Major Jam (4+)", y_test_np >= 4),
]

for name, mask in severity_buckets:
    total_true = int(np.sum(mask))
    caught = int(np.sum(strict_alerts & mask))
    bucket_recall = caught / total_true if total_true > 0 else 0.0
    print(f"{name}: caught={caught:,} / total={total_true:,} | recall={bucket_recall:.4f}")

print("\n--- RECALL ON LARGER JAM EVENTS ---")
for name, mask in [
    (">=2-band jams", y_test_np >= 2),
    (">=3-band jams", y_test_np >= 3),
    (">=4-band jams", y_test_np >= 4),
]:
    total_true = int(np.sum(mask))
    caught = int(np.sum(strict_alerts & mask))
    bucket_recall = caught / total_true if total_true > 0 else 0.0
    print(f"{name}: caught={caught:,} / total={total_true:,} | recall={bucket_recall:.4f}")


print("\n--- REAL JAM RATE BY PREDICTED CATEGORY ---")
for label in bin_labels:
    mask = (bins == label)
    total_pred = int(np.sum(mask))
    real_jams = int(np.sum(mask & (y_test_np > 0)))
    real_rate = real_jams / total_pred if total_pred > 0 else 0.0
    print(f"{label}: real_jams={real_jams:,} / total={total_pred:,} | real_jam_rate={real_rate:.4f}")