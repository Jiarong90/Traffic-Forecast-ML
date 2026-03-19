import pandas as pd

# 1. Load the 60MB file
df = pd.read_parquet("link_neighbors_200m.parquet")

# 2. Keep ONLY the two columns the neighbor_map actually needs
# (This usually deletes 3-4 'shadow' columns you didn't know were there)
df_slim = df[['link_id', 'neighbor_link_id']].copy()

# 3. Downcast to int32 
# Standard integers take 8 bytes; int32 takes 4. 
# Since Singapore LinkIDs are 9 digits, they fit perfectly in int32.
df_slim['link_id'] = df_slim['link_id'].astype('int32')
df_slim['neighbor_link_id'] = df_slim['neighbor_link_id'].astype('int32')

# 4. Use "Brotli" Compression (The gold standard for shrinking data)
# Default Parquet uses 'snappy' (fast but fat). Brotli is 'slow but tiny'.
df_slim.to_parquet(
    "link_neighbors_slim.parquet", 
    engine='pyarrow', 
    compression='brotli',
    index=False # Removes the hidden index column
)

import os
size = os.path.getsize("link_neighbors_slim.parquet") / (1024 * 1024)
print(f"💎 100% Data Kept. New File Size: {size:.2f} MB")