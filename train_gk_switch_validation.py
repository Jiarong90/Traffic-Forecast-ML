import numpy as np
import pandas as pd
import duckdb
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import gc

# -- SWITCH MODEL WITH SIMULATED DATA FROM GATEKEEPER --
# OK, since throwing majority of the engineered features at the model did not
# enable it to become more accurate, try to pivot to a new strategy
# Confirmed that the model is able to predict changes at 80% accuracy
# Now we try to see if it can predict direction of change

FEAT = "build_features_final/master_features_v3.parquet"
con = duckdb.connect()
INC_NEAR = "incident_nearby_buckets_200m.parquet"
ROAD_LINKS = "road_links.parquet"
GK_PARQUET = "gatekeeper_test_predictions.parquet"

# Fetch earliest and oldest ts record from master feature table
min_ts, max_ts = con.execute(f"""
    SELECT MIN(snapshot_ts), MAX(snapshot_ts)
    FROM read_parquet('{FEAT}')
""").fetchone()

# Split the data chronologically, 80-20 train-test split
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
      AND f.sb != f.y_tp15 -- FILTER OUT ROWS THAT DOESN'T CHANGE
      AND random() < {TRAIN_SAMPLE_FRAC}
""").df()

test_df = con.execute(f"""
    SELECT
        f.*,
        r.start_lat, r.end_lat, r.start_lon, r.end_lon,
        COALESCE(i.incident_nearby, 0) AS incident_nearby,
        COALESCE(i.nearby_accident, 0) AS nearby_accident,
        COALESCE(i.nearby_roadwork, 0) AS nearby_roadwork,
        COALESCE(i.nearby_breakdown, 0) AS nearby_breakdown,
        COALESCE(i.mins_since_nearby_start, -1) AS mins_since_nearby_start,
        g.true_change,
        g.true_down,
        g.true_up,
        g.gk_prob,
        g.gk_pred_25,
        g.gk_pred_30,
        g.gk_pred_35,
        g.gk_pred_40,
        g.is_fp,
        g.is_tp
    FROM read_parquet('{FEAT}') f
    INNER JOIN read_parquet('{GK_PARQUET}') g
      ON f.link_id = g.link_id AND f.snapshot_ts = g.snapshot_ts
    LEFT JOIN read_parquet('{ROAD_LINKS}') r
      ON f.link_id = r.link_id
    LEFT JOIN read_parquet('{INC_NEAR}') i
      ON f.link_id = i.link_id AND f.snapshot_ts = i.snapshot_ts
    WHERE g.gk_pred_25 = 1
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


# LABELS 
# Change the logic, now we want to predict direction of change
# 1 - traffic is slowing
# 2 - traffic is recovering
y_train = (train_df["y_tp15"] < train_df["sb"]).astype(np.int32)
y_test  = (test_df["y_tp15"] < test_df["sb"]).astype(np.int32)

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

# Get back some memory
del train_df
del X_train_np
del dtrain
gc.collect()

pred_prob = bst.predict(dtest)
pred = (pred_prob >= 0.5).astype(np.int32)

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

# Print accuracy scores, precision, recall, f1
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

real_change_mask = test_df["true_change"].to_numpy(dtype=np.int8) == 1

router_acc_changed = accuracy_score(y_true[real_change_mask], y_pred[real_change_mask])
router_f1_changed = f1_score(y_true[real_change_mask], y_pred[real_change_mask], zero_division=0)

print("Passed rows:", len(test_df))
print("Passed true-change rate:", test_df["true_change"].mean())
print("Passed false-positive count:", (test_df["true_change"] == 0).sum())
print("Avg GK prob on passed rows:", test_df["gk_prob"].mean())

# Check if results fall close to threshold
test_df["router_prob"] = pred_prob.astype(np.float32)
test_df["router_pred"] = (pred_prob >= 0.5).astype(np.int8)
test_df["router_correct"] = (test_df["router_pred"].to_numpy() == y_test_np).astype(np.int8)



mid_mask = (test_df["router_prob"] >= 0.45) & (test_df["router_prob"] <= 0.55)
low_mask = (test_df["router_prob"] < 0.45)
high_mask = (test_df["router_prob"] > 0.55)

print("Router acc on real changed rows:", router_acc_changed)
print("Router f1 on real changed rows:", router_f1_changed)
test_df["router_margin"] = np.abs(test_df["router_prob"] - 0.5)

for name, mask in [
    ("VERY_UNCERTAIN", test_df["router_margin"] <= 0.05),
    ("MEDIUM", (test_df["router_margin"] > 0.05) & (test_df["router_margin"] <= 0.20)),
    ("CONFIDENT", test_df["router_margin"] > 0.20),
]:
    if mask.any():
        print(f"\n{name} rows: {int(mask.sum())}")
        print("Accuracy:", test_df.loc[mask, "router_correct"].mean())
        print("True-change rate:", test_df.loc[mask, "true_change"].mean())


for name, mask in [("LOW", low_mask), ("MID", mid_mask), ("HIGH", high_mask)]:
    if mask.any():
        print(f"\n{name} confidence rows: {mask.sum()}")
        print("Accuracy:", test_df.loc[mask, "router_correct"].mean())
        if "true_change" in test_df.columns:
            print("True-change rate:", test_df.loc[mask, "true_change"].mean())

# cols_to_save = [
#     "link_id", "snapshot_ts", "sb", "y_tp15",
#     "true_change", "true_down", "true_up",
#     "gk_prob", "gk_pred_25", "gk_pred_30", "gk_pred_35", "gk_pred_40",
#     "router_prob", "router_pred", "router_correct"
# ] + features

# cols_to_save = list(dict.fromkeys(cols_to_save))


# test_df[cols_to_save].to_parquet("router_predictions.parquet", index=False)
# print("Saved router pipeline parquet:", "router_predictions.parquet")

bst.save_model("router.json")
print("Router model saved as model/router.json")

bst.feature_names = features
xgb.plot_importance(bst, max_num_features=15)
plt.show()