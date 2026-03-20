import numpy as np
import pandas as pd
import sqlite3
from sklearn.neighbors import BallTree

DB_PATH = r"D:\FYP\DB Backup\9 Feb\trafficdata.db"
INC_CLEAN = r"incidents_cleaned.parquet"
NEIGH = r"link_neighbors_200m.parquet"


# Do link_neighbors_200m to get the nearby neighboring links
# Then map incidents to nearest affected links, then get its neighbors for the "area effect"
# Then bucketize it to 5m intervals for easier querying

K = 5
ANCHOR_CUTOFF_M = 200
EARTH_R = 6371000.0

print("Loading incidents...")
inc = pd.read_parquet(INC_CLEAN).copy()

# Convert to unix seconds
inc["start_ts"] = (pd.to_datetime(inc["start_time_utc"], utc=True).astype("int64") // 1_000_000_000).astype(np.int64)
inc["end_ts"]   = (pd.to_datetime(inc["end_time_utc"],   utc=True).astype("int64") // 1_000_000_000).astype(np.int64)

print("Loading road links...")
conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
road = pd.read_sql_query("""
  SELECT link_id, start_lat, start_lon, end_lat, end_lon
  FROM road_links
""", conn)
conn.close()

road["mid_lat"] = (road["start_lat"] + road["end_lat"]) / 2.0
road["mid_lon"] = (road["start_lon"] + road["end_lon"]) / 2.0

road_coords = np.radians(road[["mid_lat","mid_lon"]].to_numpy())
inc_coords  = np.radians(inc[["lat","lon"]].to_numpy())

print("Building BallTree...")
tree = BallTree(road_coords, metric="haversine")

print(f"Finding top-{K} anchors per incident...")
dist, idx = tree.query(inc_coords, k=K)
dist_m = dist * EARTH_R
link_ids = road["link_id"].to_numpy()

# anchors - list of incident rows, and affected link ids
anchors = []
for i in range(len(inc)):
    for j in range(K):
        if dist_m[i, j] <= ANCHOR_CUTOFF_M:
            anchors.append((i, int(link_ids[idx[i, j]])))

anchors_df = pd.DataFrame(anchors, columns=["inc_row", "anchor_link_id"])
print("Total anchors:", len(anchors_df))
print("Avg anchors/incident:", len(anchors_df)/max(len(inc),1))

# Then, get neighbors of all anchors from the neighbors parquet file
print("Loading neighbors...")
neighbors = pd.read_parquet(NEIGH)[["link_id","neighbor_link_id"]]
neighbor_map = neighbors.groupby("link_id")["neighbor_link_id"].apply(lambda s: s.to_numpy()).to_dict()

# Loop the anchors and get the nearby neighbor impacted links from neighbors parquet file
print("Expand anchors and get neighbor impacted links...")
rows = []
for inc_row, grp in anchors_df.groupby("inc_row"):
    start_ts = int(inc.loc[inc_row, "start_ts"])
    end_ts   = int(inc.loc[inc_row, "end_ts"])
    itype    = inc.loc[inc_row, "type"]

    anchors_list = grp["anchor_link_id"].astype(int).tolist()

    impacted = set(anchors_list)
    for a in anchors_list:
        neigh = neighbor_map.get(a)
        if neigh is not None:
            impacted.update(neigh.astype(int).tolist())

    for link in impacted:
        rows.append((int(link), start_ts, end_ts, itype))

impact = pd.DataFrame(rows, columns=["link_id","start_ts","end_ts","type"])
impact.to_parquet("incident_impacted_links_200m.parquet", index=False)
print("Wrote incident_impacted_links_200m.parquet")
print("Rows:", len(impact))