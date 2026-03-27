import os
import duckdb

SPEED_GLOB = r"C:\Users\Admin\Desktop\Sugarcane\UNI\Y2 Sem 1\FYP\Traffic Forecast Project\Data\parquet_exports\speedbands_by_chunk\speedbands_chunk_*.parquet"
ROAD_LINKS = r"C:\Users\Admin\Desktop\Sugarcane\UNI\Y2 Sem 1\FYP\Traffic Forecast Project\Data\parquet_exports\road_links.parquet"
INCIDENTS = r"C:\Users\Admin\Desktop\Sugarcane\UNI\Y2 Sem 1\FYP\Traffic Forecast Project\ml\incidents_road_mapped.parquet"
ROAD_LINKS = r"C:/Users/Admin/Desktop/Sugarcane/UNI/Y2 Sem 1/FYP/Traffic Forecast Project/ml/road_links.parquet"
OUT_DIR = r"C:\Users\Admin\Desktop\Sugarcane\UNI\Y2 Sem 1\FYP\Traffic Forecast Project\Data\ml_datasets"
os.makedirs(OUT_DIR, exist_ok=True)

DAYS = 21
OUT = os.path.join(OUT_DIR, f"features_{DAYS}d_inc_v2_tp15.parquet")

START_TS = 1770040500
END_TS = START_TS + DAYS * 86400

OUT_SQL = OUT.replace("\\", "/")
SPEED_SQL = SPEED_GLOB.replace("\\", "/")
INC_SQL = INCIDENTS.replace("\\", "/")
ROAD_SQL = ROAD_LINKS.replace("\\", "/")

BIN_SCALE = 200

con = duckdb.connect()
con.execute("PRAGMA threads=4")
con.execute("PRAGMA preserve_insertion_order=false")
con.execute(r"PRAGMA temp_directory='D:\FYP\duckdb_temp'")

print("Building 21-day feature table...")

con.execute(f"""
COPY (
    WITH traffic AS (
        SELECT
            link_id,
            snapshot_ts,
            speed_band AS sb,
            LAG(speed_band, 1) OVER w AS sb_tm5,
            LAG(speed_band, 2) OVER w AS sb_tm10,
            LAG(speed_band, 3) OVER w AS sb_tm15,
            EXTRACT('hour' FROM (to_timestamp(snapshot_ts) + INTERVAL '8 hours')) AS hour_sg,
            EXTRACT('dow'  FROM (to_timestamp(snapshot_ts) + INTERVAL '8 hours')) AS dow_sg,
            LEAD(speed_band, 3) OVER w AS y_tp15
        FROM read_parquet('{SPEED_SQL}')
        WHERE snapshot_ts BETWEEN {START_TS} AND {END_TS}
        WINDOW w AS (PARTITION BY link_id ORDER BY snapshot_ts)
    ),
    -- FIX 1: Filter immediately to throw away useless rows BEFORE the heavy joins
    filtered_traffic AS (
        SELECT * FROM traffic
        WHERE sb BETWEEN 1 AND 8
          AND sb_tm5 BETWEEN 1 AND 8
          AND sb_tm10 BETWEEN 1 AND 8
          AND sb_tm15 BETWEEN 1 AND 8
          AND y_tp15 BETWEEN 1 AND 8
    ),
    inc AS (
        SELECT
            nearest_link_id AS link_id,
            epoch(CAST(start_time_utc AS TIMESTAMP)) AS start_ts,
            epoch(CAST(end_time_utc AS TIMESTAMP)) AS end_ts,
            type
        FROM read_parquet('{INC_SQL}')
        WHERE nearest_link_id IS NOT NULL
    ),
    joined AS (
        SELECT
            t.*,
            r.road_category,
            ((r.start_lat + r.end_lat) / 2.0) AS mid_lat,
            ((r.start_lon + r.end_lon) / 2.0) AS mid_lon,
            floor(((r.start_lat + r.end_lat) / 2.0) * {BIN_SCALE}) AS lat_bin,
            floor(((r.start_lon + r.end_lon) / 2.0) * {BIN_SCALE}) AS lon_bin,
            inc.type AS inc_type,
            inc.start_ts AS inc_start_ts
        FROM filtered_traffic t
        JOIN read_parquet('{ROAD_SQL}') r
          ON t.link_id = r.link_id
        LEFT JOIN inc
          ON inc.link_id = t.link_id
         AND t.snapshot_ts BETWEEN inc.start_ts AND inc.end_ts
    )
    SELECT
        link_id,
        snapshot_ts,
        sb, sb_tm5, sb_tm10, sb_tm15,
        hour_sg, dow_sg,
        y_tp15,
        road_category,
        mid_lat, mid_lon, lat_bin, lon_bin,

        CASE WHEN COUNT(inc_type) > 0 THEN 1 ELSE 0 END AS incident_on_link,
        COUNT(inc_type) AS incident_count_on_link,

        CASE
          WHEN MAX(inc_start_ts) IS NULL THEN NULL
          ELSE (snapshot_ts - MAX(inc_start_ts)) / 60.0
        END AS mins_since_inc_start,

        MAX(CASE WHEN inc_type = 'Accident' THEN 1 ELSE 0 END) AS has_accident,
        MAX(CASE WHEN inc_type = 'Roadwork' THEN 1 ELSE 0 END) AS has_roadwork,
        MAX(CASE WHEN inc_type = 'Vehicle breakdown' THEN 1 ELSE 0 END) AS has_breakdown

    FROM joined
    GROUP BY
        link_id, snapshot_ts,
        sb, sb_tm5, sb_tm10, sb_tm15,
        hour_sg, dow_sg, y_tp15,
        road_category, mid_lat, mid_lon, lat_bin, lon_bin
)
TO '{OUT_SQL}' (FORMAT PARQUET);
""")

print("Wrote:", OUT)