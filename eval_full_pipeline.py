import numpy as np
import pandas as pd
import duckdb
import xgboost as xgb

# PATHS
FEAT = "build_features_final/master_features_v3.parquet"
ROAD_LINKS = "road_links.parquet"
INC_NEAR = "incident_nearby_buckets_200m.parquet"
GK_PARQUET = "gatekeeper_test_predictions.parquet"
ROUTER_PARQUET = "router_predictions.parquet"
DESC_MODEL = "descent_specialist.json"
ASC_MODEL = "asc_specialist.json"  


STRICT_THRESHOLD = 0.75
TEST_SAMPLE_FRAC = 0.10


# LOAD FULL HELD-OUT TEST SET

con = duckdb.connect()

min_ts, max_ts = con.execute(f"""
    SELECT MIN(snapshot_ts), MAX(snapshot_ts)
    FROM read_parquet('{FEAT}')
""").fetchone()

split_ts = min_ts + int(0.8 * (max_ts - min_ts))
print("split_ts:", split_ts)

full_test_df = con.execute(f"""
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
        AND random() < {TEST_SAMPLE_FRAC}
""").df()

print("Full held-out rows:", len(full_test_df))


# Load the stored predictions from the first two models

gk_df = pd.read_parquet(GK_PARQUET)
router_df = pd.read_parquet(ROUTER_PARQUET)

print("GK cols:", gk_df.columns.tolist())
print("Router cols:", router_df.columns.tolist())

# Keep only needed columns
gk_keep = [c for c in ["link_id", "snapshot_ts", "gk_pred_25", "gk_prob"] if c in gk_df.columns]
router_keep = [c for c in ["link_id", "snapshot_ts", "router_pred", "router_prob"] if c in router_df.columns]

gk_df = gk_df[gk_keep].drop_duplicates(subset=["link_id", "snapshot_ts"])
router_df = router_df[router_keep].drop_duplicates(subset=["link_id", "snapshot_ts"])

# Merge the Router results
full_test_df = full_test_df.merge(
    gk_df,
    on=["link_id", "snapshot_ts"],
    how="left"
)

full_test_df = full_test_df.merge(
    router_df,
    on=["link_id", "snapshot_ts"],
    how="left"
)

# If a row has no prediction, assume no change
full_test_df["gk_pred_25"] = full_test_df["gk_pred_25"].fillna(0).astype(np.int8)
full_test_df["router_pred"] = full_test_df["router_pred"].fillna(0).astype(np.int8)

# Fill mssing probabilities with 0
if "gk_prob" in full_test_df.columns:
    full_test_df["gk_prob"] = full_test_df["gk_prob"].fillna(0.0).astype(np.float32)
if "router_prob" in full_test_df.columns:
    full_test_df["router_prob"] = full_test_df["router_prob"].fillna(0.0).astype(np.float32)

print("GK pass rows:", int((full_test_df["gk_pred_25"] == 1).sum()))
print("Router pass rows:", int((full_test_df["router_pred"] == 1).sum()))
print("Both-pass rows:", int(((full_test_df["gk_pred_25"] == 1) & (full_test_df["router_pred"] == 1)).sum()))


# FEATURE ENGINEERING

full_test_df["road_category"] = full_test_df["road_category"].astype(int)

full_test_df["delta_0_5"] = full_test_df["sb"] - full_test_df["sb_tm5"]
full_test_df["delta_5_10"] = full_test_df["sb_tm5"] - full_test_df["sb_tm10"]
full_test_df["delta_10_15"] = full_test_df["sb_tm10"] - full_test_df["sb_tm15"]
full_test_df["acceleration"] = full_test_df["delta_0_5"] - full_test_df["delta_5_10"]

full_test_df["link_dist_proxy"] = np.sqrt(
    (full_test_df["start_lat"] - full_test_df["end_lat"]) ** 2 +
    (full_test_df["start_lon"] - full_test_df["end_lon"]) ** 2
)

# Keep same convention as training file
full_test_df["is_weekend"] = full_test_df["dow_sg"].isin([0, 6]).astype(int)
full_test_df["is_peak"] = full_test_df["hour_sg"].isin([7, 8, 9, 17, 18, 19]).astype(int)

# List of the exact features the models need
features = [
    "sb","sb_tm5","sb_tm10","sb_tm15",
    "delta_0_5","delta_5_10","delta_10_15",
    "mid_lat", "mid_lon",
    "acceleration", "link_dist_proxy",
    "rain_mm", "is_raining",
    "road_category","is_weekend","is_peak",
    "incident_nearby","mins_since_nearby_start",
    "nearby_accident","nearby_roadwork","nearby_breakdown"
]

missing = [c for c in features if c not in full_test_df.columns]
print("Missing features:", missing)
if missing:
    raise ValueError(f"Missing features: {missing}")


# RUN DESCENT SPECIALIST ONLY ON ROUTED ROWS
# Create a mask for only rows where GK said Change and Router said Jam
route_mask = (full_test_df["gk_pred_25"] == 1) & (full_test_df["router_pred"] == 1)

full_test_df["pred_mag"] = 0.0
full_test_df["strict_alert"] = 0

if route_mask.any():
    X_routed = full_test_df.loc[route_mask, features].astype(np.float32)
    dtest = xgb.DMatrix(X_routed, feature_names=features)

    # Load the Descent model and predict the magnitude of the drop
    bst = xgb.Booster()
    bst.load_model(DESC_MODEL)

    pred_mag = bst.predict(dtest)
    pred_mag = np.clip(pred_mag, 0, None)

    full_test_df.loc[route_mask, "pred_mag"] = pred_mag
    full_test_df.loc[route_mask, "strict_alert"] = (pred_mag >= STRICT_THRESHOLD).astype(np.int8)

print("Final alert rows:", int((full_test_df["strict_alert"] == 1).sum()))


# FULL-PIPELINE METRICS

# Calcualte real-world jam severity (current speed - future speed)
true_jam_mag = np.maximum(full_test_df["sb"].to_numpy() - full_test_df["y_tp15"].to_numpy(), 0)
strict_alert = full_test_df["strict_alert"].to_numpy().astype(bool)

print("\n=== FULL PIPELINE RECALL ON TRUE JAM SEVERITY ===")
# Loop through different jam to see which ones were caught
for name, mask in [
    ("1-band jams", true_jam_mag == 1),
    ("2-3 band jams", (true_jam_mag >= 2) & (true_jam_mag <= 3)),
    ("4+ band jams", true_jam_mag >= 4),
]:
    total_true = int(np.sum(mask))
    caught = int(np.sum(strict_alert & mask))
    recall = caught / total_true if total_true > 0 else 0.0
    print(f"{name}: caught={caught:,} / total={total_true:,} | recall={recall:.4f}")

# Re-run for cumulative categories
print("\n=== FULL PIPELINE RECALL ON LARGE JAMS ===")
for name, mask in [
    (">=2-band jams", true_jam_mag >= 2),
    (">=3-band jams", true_jam_mag >= 3),
    (">=4-band jams", true_jam_mag >= 4),
]:
    total_true = int(np.sum(mask))
    caught = int(np.sum(strict_alert & mask))
    recall = caught / total_true if total_true > 0 else 0.0
    print(f"{name}: caught={caught:,} / total={total_true:,} | overall_pipeline_recall={recall:.4f}")

# STAGE-BY-STAGE RECALL
gk_mask = (full_test_df["gk_pred_25"].to_numpy() == 1)
router_mask = ((full_test_df["gk_pred_25"].to_numpy() == 1) & (full_test_df["router_pred"].to_numpy() == 1))

print("\n=== STAGE-BY-STAGE RECALL ON LARGE JAMS ===")
for name, mask in [
    (">=2-band jams", true_jam_mag >= 2),
    (">=3-band jams", true_jam_mag >= 3),
    (">=4-band jams", true_jam_mag >= 4),
]:
    total_true = int(np.sum(mask))
    gk_caught = int(np.sum(gk_mask & mask))
    routed_caught = int(np.sum(router_mask & mask))
    final_caught = int(np.sum(strict_alert & mask))

    gk_recall = gk_caught / total_true if total_true > 0 else 0.0
    routed_recall = routed_caught / total_true if total_true > 0 else 0.0
    final_recall = final_caught / total_true if total_true > 0 else 0.0

    print(f"{name}:")
    print(f"  gatekeeper recall = {gk_caught:,}/{total_true:,} = {gk_recall:.4f}")
    print(f"  after router      = {routed_caught:,}/{total_true:,} = {routed_recall:.4f}")
    print(f"  final pipeline    = {final_caught:,}/{total_true:,} = {final_recall:.4f}")




print("FOR ASCENT")



route_mask_asc = (full_test_df["gk_pred_25"] == 1) & (full_test_df["router_pred"] == 0)

full_test_df["pred_mag_asc"] = 0.0
full_test_df["strict_alert_asc"] = 0

if route_mask_asc.any():
    X_routed_asc = full_test_df.loc[route_mask_asc, features].astype(np.float32)
    dtest_asc = xgb.DMatrix(X_routed_asc, feature_names=features)

    bst_asc = xgb.Booster()
    bst_asc.load_model(ASC_MODEL)

    pred_mag_asc = bst_asc.predict(dtest_asc)
    pred_mag_asc = np.clip(pred_mag_asc, 0, None)

    full_test_df.loc[route_mask_asc, "pred_mag_asc"] = pred_mag_asc
    full_test_df.loc[route_mask_asc, "strict_alert_asc"] = (
        pred_mag_asc >= STRICT_THRESHOLD
    ).astype(np.int8)

strict_alert_asc = full_test_df["strict_alert_asc"].to_numpy().astype(bool)

true_recovery_mag = np.maximum(
    full_test_df["y_tp15"].to_numpy() - full_test_df["sb"].to_numpy(), 0
)

print("\n=== ASCENT: FULL PIPELINE RECALL ON TRUE RECOVERY ===")
for name, mask in [
    ("1-band recovery",   true_recovery_mag == 1),
    ("2-3 band recovery", (true_recovery_mag >= 2) & (true_recovery_mag <= 3)),
    ("4+ band recovery",  true_recovery_mag >= 4),
]:
    total_true = int(np.sum(mask))
    caught     = int(np.sum(strict_alert_asc & mask))
    recall     = caught / total_true if total_true > 0 else 0.0
    print(f"{name}: caught={caught:,} / total={total_true:,} | recall={recall:.4f}")

print("\n=== ASCENT: STAGE-BY-STAGE RECALL ===")
for name, mask in [
    (">=2-band recovery", true_recovery_mag >= 2),
    (">=3-band recovery", true_recovery_mag >= 3),
    (">=4-band recovery", true_recovery_mag >= 4),
]:
    total_true    = int(np.sum(mask))
    gk_caught     = int(np.sum(gk_mask & mask))
    routed_caught = int(np.sum(route_mask_asc.to_numpy() & mask))
    final_caught  = int(np.sum(strict_alert_asc & mask))

    print(f"{name}:")
    print(f"  gatekeeper recall = {gk_caught:,}/{total_true:,} = {gk_caught/total_true:.4f}")
    print(f"  after router      = {routed_caught:,}/{total_true:,} = {routed_caught/total_true:.4f}")
    print(f"  final pipeline    = {final_caught:,}/{total_true:,} = {final_caught/total_true:.4f}")




def smoke_big_deltas(df, mag_col, title, threshold=2.0):
    print(f"\n{'='*40}")
    print(f"SCENARIOS WITH PREDICTED CHANGE >= {threshold}")
    print(f"{'='*40}")
    
    # Filter for cases where the predicted magnitude is large
    big_changes = df[df[mag_col] >= threshold].sort_values(by=mag_col, ascending=False).head(5)
    
    if big_changes.empty:
        print(f"No massive changes found in this sample for {title}.")
        return

    for i, (idx, row) in enumerate(big_changes.iterrows()):
        pred_val = row[mag_col]
        print(f"\n[Case #{i+1}] PREDICTED DROP: -{pred_val:.2f} Bands" if "DESCENT" in title else f"\n[Case #{i+1}] PREDICTED RISE: +{pred_val:.2f} Bands")
        print(f"Current SB (Input): {int(row['sb'])}  --->  Predicted SB (T+15): {round(row['sb'] - pred_val if 'DESCENT' in title else row['sb'] + pred_val)}")
        print("-" * 40)
        print(">>> TYPE THESE INTO YOUR UI:")
        print(f"  Speed Now: {int(row['sb'])}")
        print(f"  History: T-5={int(row['sb_tm5'])}, T-10={int(row['sb_tm10'])}, T-15={int(row['sb_tm15'])}")
        print(f"  Incident: Any={int(row['incident_nearby'])}, Accident={int(row['nearby_accident'])}")
        print(f"  Timer: Mins Since Start={int(row['mins_since_nearby_start'])}")
        print(f"  Rain: {row['rain_mm']}mm | Peak: {int(row['is_peak'])}")
        print("-" * 40)

# Smoke the big predicted drops (Jams)
smoke_big_deltas(full_test_df[full_test_df["router_pred"] == 1], "pred_mag", "MASSIVE JAMS")

# Smoke the big predicted rises (Recoveries)
smoke_big_deltas(full_test_df[full_test_df["router_pred"] == 0], "pred_mag_asc", "MASSIVE RECOVERIES")





def smoke_jams(df, n=3):
    print(f"\n{'='*40}")
    print(f"SCENARIOS FOR MASSIVE JAMS (SPEED DROPS)")
    print(f"{'='*40}")
    
    # Sort by 'pred_mag' to find the biggest drops
    top_jams = df.sort_values(by="pred_mag", ascending=False).head(n)
    
    if top_jams.empty:
        print("No massive jams found")
        return

    for i, (idx, row) in enumerate(top_jams.iterrows()):
        drop_amount = row['pred_mag']
        current_sb = int(row['sb'])
        predicted_sb = max(1, round(current_sb - drop_amount))
        
        print(f"\n[JAM Case #{i+1}] PREDICTED DROP: -{drop_amount:.2f} Bands")
        print(f"Current SB: {current_sb} >  Predicted SB (T+15): {predicted_sb}")
        print("-" * 40)
        print(f"  Current Speedband: {current_sb}")
        print(f"  T-5: {int(row['sb_tm5'])} | T-10: {int(row['sb_tm10'])} | T-15: {int(row['sb_tm15'])}")
        print(f"  Incident Nearby: {int(row['incident_nearby'])}")
        print(f"  Accident: {int(row['nearby_accident'])}")
        print(f"  Mins Since Start: {int(row['mins_since_nearby_start'])}")
        print(f"  Rain: {row['rain_mm']} | Peak: {int(row['is_peak'])}")
        print("-" * 40)

# Call it using the mask for the Descent specialist (router_pred == 1)
smoke_jams(full_test_df[full_test_df["router_pred"] == 1])