import requests
import sqlite3

DB_PATH = r"D:\FYP\DB Backup\9 Feb\trafficdata.db"
URL = "https://api-open.data.gov.sg/v2/real-time/api/rainfall"

def rainfall_stations():
    print("Fetching station data from API...")
    response = requests.get(URL, timeout=30)
    response.raise_for_status()
    data = response.json()
    
    stations = data["data"]["stations"]
    
    print(f"Connecting to {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    updated_count = 0
    
    for station in stations:
        station_id = station["id"]
        loc = station.get("location")
        
        if loc and loc.get("latitude") and loc.get("longitude"):
            lat = float(loc.get("latitude"))
            lon = float(loc.get("longitude"))
            
            cur.execute("""
                UPDATE rainfall_stations 
                SET lat = ?, lon = ? 
                WHERE station_id = ?
            """, (lat, lon, station_id))
            
            if cur.rowcount > 0:
                updated_count += 1
            
    conn.commit()
    conn.close()
    print(f"Successfully updated {updated_count} stations with coordinates!")

if __name__ == "__main__":
    rainfall_stations()