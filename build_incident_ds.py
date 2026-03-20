import requests
import os
import json
import sqlite3
import datetime
import time
import pandas as pd
from math import radians, sin, cos, sqrt, atan2


DB_PATH = r"D:\FYP\DB Backup\9 Feb\trafficdata.db"

# Thresholds
DIST_THRESHOLD = 200
MAX_GAP = 15

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    dlat = radians(lat2- lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))

print("Connecting to DB (read-only)..")
conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)

query = """
    SELECT type, lat, lon, message, snapshot_time
    FROM traffic_incidents
    ORDER BY snapshot_time
"""

incidents = pd.read_sql_query(query, conn)
print(f"Loaded rows: {len(incidents)}")
print(incidents.head())

incidents["ts"] = pd.to_datetime(incidents["snapshot_time"], utc=True)
confirmed_incidents = []
occuring_incidents = []
next_id = 1

for row in incidents.itertuples(index=False):
    best_idx = None
    best_dist = None

    for i, incident in enumerate(occuring_incidents):
        if incident["type"] != row.type:
            continue

        gap_min = (row.ts - incident["last_seen"]).total_seconds() / 60.0
        if gap_min > MAX_GAP:
            continue

        dist = haversine_m(row.lat, row.lon, incident["lat"], incident["lon"])
        if dist <= DIST_THRESHOLD and (best_dist is None or dist < best_dist):
            best_dist = dist
            best_idx = i

    if best_dist is None:
        occuring_incidents.append({
            "incident_id": next_id,
            "type": row.type,
            "lat": float(row.lat),
            "lon": float(row.lon),
            "start": row.ts,
            "end": row.ts,
            "last_seen": row.ts,
            "message": row.message if isinstance(row.message, str) else None,
        })
        next_id += 1

    else:
        incident = occuring_incidents[best_idx]
        incident["end"] = row.ts
        incident["last_seen"] = row.ts

    incidents_still_occuring = []
    for incident in occuring_incidents:
        gap_min = (row.ts - incident["last_seen"]).total_seconds() / 60.0
        if gap_min > MAX_GAP:
            duration_min = (incident["end"] - incident["start"]).total_seconds() / 60.0
            confirmed_incidents.append({
                "incident_id": incident["incident_id"],
                "type": incident["type"],
                "lat": incident["lat"],
                "lon": incident["lon"],
                "start_time_utc": incident["start"].isoformat(),
                "end_time_utc": incident["end"].isoformat(),
                "duration_min": duration_min,
                "message": incident["message"],
            })
        else:
            incidents_still_occuring.append(incident)
    occuring_incidents = incidents_still_occuring

for incident in occuring_incidents:
    duration_min = (incident["end"] - incident["start"]).total_seconds() / 60.0
    confirmed_incidents.append({
        "incident_id": incident["incident_id"],
        "type": incident["type"],
        "lat": incident["lat"],
        "lon": incident["lon"],
        "start_time_utc": incident["start"].isoformat(),
        "end_time_utc": incident["end"].isoformat(),
        "duration_min": duration_min,
        "message": incident["message"],
    })

incidents_clean = pd.DataFrame(confirmed_incidents).sort_values("start_time_utc").reset_index(drop=True)

print("Snapshot rows: ", len(incidents))
print("Cleaned Incidents: ", len(incidents_clean))
print(incidents_clean.head())



print("\nIncident types distribution:")
print(incidents_clean["type"].value_counts())

out_path = "incidents_cleaned.parquet"
incidents_clean.to_parquet(out_path, index=False)





conn.close()



