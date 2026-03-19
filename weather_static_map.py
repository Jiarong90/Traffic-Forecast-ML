import pandas as pd
from sklearn.neighbors import BallTree
import numpy as np
import sqlite3
DB_PATH = r"D:\FYP\DB Backup\9 Feb\trafficdata.db"

# Load your link coordinates (from your main dataset)
conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)

with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as conn:
    stations = pd.read_sql("SELECT station_id, lat, lon FROM rainfall_stations", conn)
links = pd.read_parquet("road_links.parquet") 
# Get the lat and lon from stations

# Convert to Radians, as Haversine is a trigonometric formula and needs radians
link_rads = np.deg2rad(links[['mid_lat', 'mid_lon']].values)
station_rads = np.deg2rad(stations[['lat', 'lon']].values)
# Build the tree using Stations
tree = BallTree(station_rads, metric='haversine')
# Get the distance and index from nearest k=1 link
dist, ind = tree.query(link_rads, k=1)

# Use the row indexes to map each road to its closest station ID
links['nearest_station_id'] = stations['station_id'].iloc[ind.flatten()].values
# Save it to Parquet
links[['link_id', 'nearest_station_id']].to_parquet("link_station_mapping.parquet")


mapping = pd.read_parquet("link_station_mapping.parquet")
links = pd.read_parquet("road_links.parquet")

check_df = links.merge(mapping, on='link_id')

# Look at 5 random examples
print(check_df[['road_name', 'nearest_station_id']].sample(5))