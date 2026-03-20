import pandas as pd
import numpy as np

IMPACT = "incident_impacted_links_200m.parquet"
OUT = "incident_nearby_buckets_200m.parquet"

# Do link_neighbors_200m to get the nearby neighboring links
# Then map incidents to nearest affected links, then get its neighbors for the "area effect"
# Then bucketize it to 5m intervals for easier querying



print("Loading impacted intervals...")
imp = pd.read_parquet(IMPACT).copy()

# snap start/end to 5-min buckets
imp["start_b"] = (imp["start_ts"] // 300) * 300
imp["end_b"]   = (imp["end_ts"]   // 300) * 300

print("Expanding to 5-min buckets (this may take a bit)...")
rows = []
for r in imp.itertuples(index=False):
    # iterate buckets inclusively
    for ts in range(int(r.start_b), int(r.end_b) + 300, 300):
        rows.append((int(r.link_id), int(ts), str(r.type), int(r.start_ts)))

df = pd.DataFrame(rows, columns=["link_id","snapshot_ts","type","start_ts"])

# aggregate if multiple incidents overlap on same link+time
out = df.groupby(["link_id","snapshot_ts"], as_index=False).agg(
    incident_nearby=("type", lambda s: 1),
    nearby_accident=("type", lambda s: int((s=="Accident").any())),
    nearby_roadwork=("type", lambda s: int((s=="Roadwork").any())),
    nearby_breakdown=("type", lambda s: int((s=="Vehicle breakdown").any())),
    most_recent_start_ts=("start_ts", "max"),
)

out["mins_since_nearby_start"] = (out["snapshot_ts"] - out["most_recent_start_ts"]) / 60.0
out = out.drop(columns=["most_recent_start_ts"])

out.to_parquet(OUT, index=False)
print("Wrote:", OUT)
print("Rows:", len(out))