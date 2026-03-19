import duckdb

DB_PATH = r"D:\FYP\DB Backup\9 Feb\trafficdata.db"
MAPPING_FILE = "link_station_mapping.parquet"

con = duckdb.connect()

con.execute(f"INSTALL sqlite; LOAD sqlite;")
con.execute(f"ATTACH '{DB_PATH}' AS sqlite_db (TYPE SQLITE)")

con.execute(f"CREATE VIEW mapping AS SELECT * FROM read_parquet('{MAPPING_FILE}')")

# Create the Weather Features table
# Join the mapping to the actual readings based on station_id
print("Mapping rainfall readings to road links...")
con.execute("""
    CREATE OR REPLACE TABLE weather_features AS
    SELECT 
        m.link_id,
        r.reading_time,
        r.value_mm AS rain_mm,
        CASE WHEN r.value_mm > 0 THEN 1 ELSE 0 END AS is_raining
    FROM mapping m
    JOIN sqlite_db.rainfall r ON m.nearest_station_id = r.station_id
""")

# Export to Parquet
con.execute("COPY weather_features TO 'weather_features.parquet' (FORMAT PARQUET)")
print("Successfully created weather_features.parquet")



