import os, sqlite3, pandas as pd

DB_PATH = r"C:\Users\Admin\Desktop\Sugarcane\UNI\Y2 Sem 1\FYP\Traffic Forecast Project\Data\trafficdata.db"
OUT_PATH = r"C:\Users\Admin\Desktop\Sugarcane\UNI\Y2 Sem 1\FYP\Traffic Forecast Project\ml\road_links.parquet"

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
df = pd.read_sql_query("""
    SELECT link_id, road_name, road_category, start_lat, start_lon, end_lat, end_lon 
    FROM road_links
""", conn)
conn.close()

df["link_id"] = pd.to_numeric(df["link_id"], downcast="integer")
df["mid_lat"] = (df["start_lat"] + df["end_lat"]) / 2
df["mid_lon"] = (df["start_lon"] + df["end_lon"]) / 2
df.to_parquet(OUT_PATH, index=False, compression="snappy")

print("Wrote", OUT_PATH, "rows: ", len(df))