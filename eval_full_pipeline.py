import numpy as np
import pandas as pd
import duckdb
import xgboost as xgb

# =========================
# PATHS
# =========================
FEAT = "build_features_final/master_features_v3.parquet"
ROAD_LINKS = "road_links.parquet"
INC_NEAR = "incident_nearby_buckets_200m.parquet"
GK_PARQUET = "gatekeeper_test_predictions.parquet"
ROUTER_PARQUET = "router_predictions.parquet"
DESC_MODEL = "descent_specialist.json"

STRICT_THRESHOLD = 0.75
TEST_SAMPLE_FRAC = 0.10

# =========================
# LOAD FULL HELD-OUT TEST SET
# =========================
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

# =========================
# JOIN GATEKEEPER + ROUTER OUTPUTS
# =========================
gk_df = pd.read_parquet(GK_PARQUET)
router_df = pd.read_parquet(ROUTER_PARQUET)

print("GK cols:", gk_df.columns.tolist())
print("Router cols:", router_df.columns.tolist())

# Keep only needed columns
gk_keep = [c for c in ["link_id", "snapshot_ts", "gk_pred_25", "gk_prob"] if c in gk_df.columns]
router_keep = [c for c in ["link_id", "snapshot_ts", "router_pred", "router_prob"] if c in router_df.columns]

gk_df = gk_df[gk_keep].drop_duplicates(subset=["link_id", "snapshot_ts"])
router_df = router_df[router_keep].drop_duplicates(subset=["link_id", "snapshot_ts"])

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

full_test_df["gk_pred_25"] = full_test_df["gk_pred_25"].fillna(0).astype(np.int8)
full_test_df["router_pred"] = full_test_df["router_pred"].fillna(0).astype(np.int8)

if "gk_prob" in full_test_df.columns:
    full_test_df["gk_prob"] = full_test_df["gk_prob"].fillna(0.0).astype(np.float32)
if "router_prob" in full_test_df.columns:
    full_test_df["router_prob"] = full_test_df["router_prob"].fillna(0.0).astype(np.float32)

print("GK pass rows:", int((full_test_df["gk_pred_25"] == 1).sum()))
print("Router pass rows:", int((full_test_df["router_pred"] == 1).sum()))
print("Both-pass rows:", int(((full_test_df["gk_pred_25"] == 1) & (full_test_df["router_pred"] == 1)).sum()))

# =========================
# FEATURE ENGINEERING
# =========================
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

# =========================
# RUN DESCENT SPECIALIST ONLY ON ROUTED ROWS
# =========================
route_mask = (full_test_df["gk_pred_25"] == 1) & (full_test_df["router_pred"] == 1)

full_test_df["pred_mag"] = 0.0
full_test_df["strict_alert"] = 0

if route_mask.any():
    X_routed = full_test_df.loc[route_mask, features].to_numpy(dtype=np.float32, copy=False)
    dtest = xgb.QuantileDMatrix(X_routed)

    bst = xgb.Booster()
    bst.load_model(DESC_MODEL)

    pred_mag = bst.predict(dtest)
    pred_mag = np.clip(pred_mag, 0, None)

    full_test_df.loc[route_mask, "pred_mag"] = pred_mag
    full_test_df.loc[route_mask, "strict_alert"] = (pred_mag >= STRICT_THRESHOLD).astype(np.int8)

print("Final alert rows:", int((full_test_df["strict_alert"] == 1).sum()))

# =========================
# FULL-PIPELINE METRICS
# =========================
true_jam_mag = np.maximum(full_test_df["sb"].to_numpy() - full_test_df["y_tp15"].to_numpy(), 0)
strict_alert = full_test_df["strict_alert"].to_numpy().astype(bool)

print("\n=== FULL PIPELINE RECALL ON TRUE JAM SEVERITY ===")
for name, mask in [
    ("1-band jams", true_jam_mag == 1),
    ("2-3 band jams", (true_jam_mag >= 2) & (true_jam_mag <= 3)),
    ("4+ band jams", true_jam_mag >= 4),
]:
    total_true = int(np.sum(mask))
    caught = int(np.sum(strict_alert & mask))
    recall = caught / total_true if total_true > 0 else 0.0
    print(f"{name}: caught={caught:,} / total={total_true:,} | recall={recall:.4f}")

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

# =========================
# OPTIONAL: STAGE-BY-STAGE RECALL
# =========================
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