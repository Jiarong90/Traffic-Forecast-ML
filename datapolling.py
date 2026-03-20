import requests
import os
import json
import sqlite3
import datetime
import time
from dotenv import load_dotenv

load_dotenv()

LTA_API_KEY = os.getenv("LTA_API_KEY")
DATA_GOV_API_KEY = os.getenv("DATA_GOV_API_KEY")

url_speedbands = "https://datamall2.mytransport.sg/ltaodataservice/v4/TrafficSpeedBands"
url_incidents = "https://datamall2.mytransport.sg/ltaodataservice/TrafficIncidents"
url_estimated_traveltime = "https://datamall2.mytransport.sg/ltaodataservice/EstTravelTimes"
url_faulty_trafficlights = "https://datamall2.mytransport.sg/ltaodataservice/FaultyTrafficLights"
url_vms_emas = "https://datamall2.mytransport.sg/ltaodataservice/VMS" 
url_train_service_alerts = "https://datamall2.mytransport.sg/ltaodataservice/TrainServiceAlerts"
url_rainfall = "https://api-open.data.gov.sg/v2/real-time/api/rainfall"

lta_headers = {
    "AccountKey": LTA_API_KEY,
    "accept": "application/json"
}

weather_headers = {
    "X-Api-Key": DATA_GOV_API_KEY,
    "accept": "application/json"
}

TIME_BASED_POLLING = True

SGT_TZ = datetime.timezone(datetime.timedelta(hours=8))
ACTIVE_START_HOUR = 4
ACTIVE_END_HOUR = 22

def in_active_window(now_utc: datetime.datetime) -> bool:
    sgt_now = now_utc.astimezone(SGT_TZ)
    hour = sgt_now.hour
    return ACTIVE_START_HOUR <= hour < ACTIVE_END_HOUR

def sleep_until_next_window(now_utc: datetime.datetime):
    sgt_now = now_utc.astimezone(SGT_TZ)
    today = sgt_now.date()
    if sgt_now.hour >= ACTIVE_END_HOUR:
        target_date = today + datetime.timedelta(days=1)
    else:
        target_date = today

    next_sgt = datetime.datetime(year=target_date.year, month=target_date.month, day=target_date.day, hour=ACTIVE_START_HOUR, minute=0, second=0, tzinfo=SGT_TZ)
    next_utc = next_sgt.astimezone(datetime.timezone.utc)
    sleep_seconds = (next_utc - now_utc).total_seconds()
    if sleep_seconds > 0:
        time.sleep(sleep_seconds + 0.1)

def sleep_until_next():
    now = datetime.datetime.now(datetime.timezone.utc)
    sec_since_hour = now.minute * 60 + now.second + now.microsecond / 1_000_000
    remainder = sec_since_hour % 300
    if remainder == 0:
        return
    wait = 300 - remainder
    time.sleep(wait + 0.1)

def get_all_lta_data(base_url, headers, page_size=500):
    results = []
    skip = 0
    while True:
        separator = "&" if "?" in base_url else "?"
        url = f"{base_url}{separator}$skip={skip}"

        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            batch = data.get("value", [])
            if not batch:
                break
            results.extend(batch)

            # If batch has fewer than page_size, end of record
            if len(batch) < page_size:
                break

            skip += page_size

        except Exception as e:
            print(f"Error fetching LTA data at {skip}: {e}")
            break

    return results

def check_int_none(val):
    if val is None or val == "":
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None

def initialize_sqlite():
    # Connect to sqlite
    conn = sqlite3.connect("trafficdata2.db")
    cur = conn.cursor()

    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")

    # Create table for road_link, store data for each link_id 
    cur.execute("""
    CREATE TABLE IF NOT EXISTS road_links (
        link_id TEXT PRIMARY KEY,
        road_name TEXT,
        road_category TEXT,
        start_lat REAL,
        start_lon REAL,
        end_lat REAL,
        end_lon REAL
    )
    """)

    # Create table for speed bands
    cur.execute("""
    CREATE TABLE IF NOT EXISTS speedbands (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        link_id TEXT,
        speed_band INTEGER,
        min_speed INTEGER,
        max_speed INTEGER,
        snapshot_time TEXT
    )
    """)

    # Create table for traffic incidents
    cur.execute("""
    CREATE TABLE IF NOT EXISTS traffic_incidents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        type TEXT,
        lat REAL,
        lon REAL,
        message TEXT,
        snapshot_time TEXT
    )
    """)

    # Create table for est travel times
    cur.execute("""
    CREATE TABLE IF NOT EXISTS est_trav_times (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        direction INTEGER,
        far_end_point TEXT,
        start_point TEXT,
        end_point TEXT,
        est_time INTEGER,
        snapshot_time TEXT
    )
    """)

    # Create table for faulty traffic lights
    cur.execute("""
    CREATE TABLE IF NOT EXISTS faulty_traffic_lights (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        alarm_id TEXT,
        node_id TEXT,
        type INTEGER,
        start_date TEXT,
        end_date TEXT,
        message TEXT,
        snapshot_time TEXT
    )
    """)

    # Create table for VMS / EMAS
    cur.execute("""
    CREATE TABLE IF NOT EXISTS VMS (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        equipment_id TEXT,
        lat REAL,
        lon REAL,
        message TEXT,
        snapshot_time TEXT
    )
    """)

    # Create table for Train Service Alerts
    cur.execute("""
    CREATE TABLE IF NOT EXISTS train_service_alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        status INTEGER,
        line TEXT,
        direction TEXT,
        stations TEXT,
        free_bus TEXT,
        free_shuttle TEXT,
        shuttle_direction TEXT,
        message TEXT,
        created_time TEXT,
        snapshot_time TEXT
    )
    """)

    # Create table for Rainfall stations
    cur.execute("""
    CREATE TABLE IF NOT EXISTS rainfall_stations (
        station_id TEXT PRIMARY KEY,
        loc_name TEXT,
        lat REAL,
        lon REAL
    )
    """)

    # Create table for Rainfall 
    cur.execute("""
    CREATE TABLE IF NOT EXISTS rainfall (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        station_id TEXT,
        value_mm REAL,
        reading_time TEXT,
        snapshot_time TEXT
    )
    """)

    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_speedbands_link_time
    ON speedbands (link_id, snapshot_time)
    """)
    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_incidents_time
    ON traffic_incidents (snapshot_time)
    """)
    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_est_name_time
    ON est_trav_times (name, snapshot_time)
    """)
    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_faulty_alarm_time
    ON faulty_traffic_lights (alarm_id, snapshot_time)
    """)
    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_vms_equipment_time
    ON VMS (equipment_id, snapshot_time)
    """)

    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_train_alert_line_time
    ON train_service_alerts (line, snapshot_time)
    """)

    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_rainfall_station_time
    ON rainfall (station_id, snapshot_time)
    """)

    cur.execute("""
    CREATE UNIQUE INDEX IF NOT EXISTS idx_rainfall_unique
    ON rainfall (station_id, reading_time)
    """)

    conn.commit()
    return conn

def get_speedbands_data():
    return get_all_lta_data(url_speedbands, lta_headers)

def get_incidents_data():
    return get_all_lta_data(url_incidents, lta_headers)

def get_estimated_tt_data():
    return get_all_lta_data(url_estimated_traveltime, lta_headers)

def get_faulty_tl_data():
    return get_all_lta_data(url_faulty_trafficlights, lta_headers)

def get_vms_data():
    return get_all_lta_data(url_vms_emas, lta_headers)

def get_train_service_alerts_data():
    response = requests.get(url_train_service_alerts, headers=lta_headers, timeout=30)
    response.raise_for_status()
    data = response.json()

    value = data.get("value")
    if value is None:
        return []
    
    if isinstance(value, list):
        return value
    
    if isinstance(value, dict):
        status = value.get("Status")
        segments = value.get("Line", []) or []
        messages = value.get("Message", []) or []
        msg_content = None
        msg_created = None

        if isinstance(messages, list) and messages:
            first_msg = messages[0]
            if isinstance(first_msg, dict):
                msg_content = first_msg.get("Content")
                msg_created = first_msg.get("CreatedDate")

        rows = []
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            rows.append({
                "Status": status,
                "Line": seg.get("Line"),
                "Direction": seg.get("Direction"),
                "Stations": seg.get("Stations"),
                "FreePublicBus": value.get("FreePublicBus"),
                "FreeMRTShuttle": value.get("FreeMRTShuttle"),
                "MRTShuttleDirection": seg.get("MRTShuttleDirection"),
                "Message": msg_content,
                "CreatedDate": msg_created,
            })

        return rows
    
    return []

def get_rainfall_data():
    response = requests.get(url_rainfall, headers=weather_headers, timeout=30)
    response.raise_for_status()
    data = response.json()
    return data


def save_speedbands_data(rows, conn, snapshot_time):
    cur = conn.cursor()

    links_data = []
    speedbands_data = []
    for item in rows:
        
        links_data.append((
            item["LinkID"],
            item.get("RoadName"),
            str(item.get("RoadCategory")),
            float(item["StartLat"]),
            float(item["StartLon"]),
            float(item["EndLat"]),
            float(item["EndLon"])
        ))

        speedbands_data.append((
            item["LinkID"],
            check_int_none(item["SpeedBand"]),
            check_int_none(item["MinimumSpeed"]),
            check_int_none(item["MaximumSpeed"]),
            snapshot_time
        ))
    if links_data:
        # Insert road link data
        # Static data, row only inserted once per link id
        cur.executemany("""
            INSERT OR IGNORE INTO road_links (
                link_id, road_name, road_category, start_lat, start_lon,
                end_lat, end_lon
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, links_data)

    if speedbands_data:
        # Insert speed bands data
        # Dynamic data, rows inserted in intervals with link id as foreign key
        cur.executemany("""
            INSERT INTO speedbands (
                link_id, speed_band, min_speed, max_speed, snapshot_time
            )
            VALUES (?, ?, ?, ?, ?)
        """, speedbands_data)

    conn.commit()

def save_incidents_data(rows, conn, snapshot_time):
    cur = conn.cursor()

    for item in rows:
        # Insert incidents data
        cur.execute("""
            INSERT INTO traffic_incidents (
                type, lat, lon, message, snapshot_time
            )
            VALUES (?, ?, ?, ?, ?)
        """, (
            item["Type"],
            float(item["Latitude"]),
            float(item["Longitude"]),
            item["Message"],
            snapshot_time
        ))
    conn.commit()

def save_estimated_tt_data(rows, conn, snapshot_time):
    cur = conn.cursor()

    for item in rows:
        est_time_val = check_int_none(item.get("EstTime"))
        # Insert estimated travel times data
        cur.execute("""
            INSERT INTO est_trav_times (
                name, direction, far_end_point, start_point, 
                end_point, est_time, snapshot_time
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            item["Name"],
            check_int_none(item.get("Direction")),
            str(item.get("FarEndPoint")),
            str(item.get("StartPoint")),
            str(item.get("EndPoint")),
            est_time_val,
            snapshot_time
        ))
    conn.commit()

def save_faulty_tl_data(rows, conn, snapshot_time):
    cur = conn.cursor()

    for item in rows:
        # Insert faulty traffic lights data
        
        alarm_id   = item["AlarmID"]
        node_id    = item["NodeID"]
        alarm_type = int(item["Type"])
        start_date = item.get("StartDate")
        end_date   = item.get("EndDate")
        message    = item.get("Message")
        cur.execute("""
            INSERT INTO faulty_traffic_lights (
                alarm_id, node_id, type, start_date, end_date, message, snapshot_time
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            alarm_id,
            node_id,
            alarm_type,
            start_date,
            end_date,
            message,
            snapshot_time
        ))
    conn.commit()

def save_vms_data(rows, conn, snapshot_time):
    cur = conn.cursor()

    for item in rows:
        # Insert VMS / EMAS data
        cur.execute("""
            INSERT INTO VMS (
                equipment_id, lat, lon, message, snapshot_time
            )
            VALUES (?, ?, ?, ?, ?)
        """, (
            item["EquipmentID"],
            float(item["Latitude"]),
            float(item["Longitude"]),
            item["Message"],
            snapshot_time
        ))
    conn.commit()

def save_train_service_alerts_data(rows, conn, snapshot_time):
    cur = conn.cursor()

    for item in rows:
        # Insert train service alerts data
       
        status_val = check_int_none(item.get("Status"))
        cur.execute("""
            INSERT INTO train_service_alerts (
                status, line, direction, stations, free_bus, free_shuttle,
                shuttle_direction, message, created_time, snapshot_time
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            status_val,
            item.get("Line"),
            item.get("Direction"),
            item.get("Stations"),
            item.get("FreePublicBus"),
            item.get("FreeMRTShuttle"),
            item.get("MRTShuttleDirection"),
            item.get("Message"),
            item.get("CreatedDate"),
            snapshot_time
        ))
    conn.commit()

def save_rainfall_stations(rows, conn):
    cur = conn.cursor()
    stations = rows["data"]["stations"]
    for station in stations:
        # Insert rainfall station data
        # Static data, row only inserted once per station id
        loc = station.get("labelLocation")
        if loc:
            lat = loc.get("latitude")
            lon = loc.get("longitude")
            lat = float(lat) if lat is not None else None
            lon = float(lon) if lon is not None else None
        else:
            lat = None
            lon = None
        cur.execute("""
            INSERT OR IGNORE INTO rainfall_stations (
                station_id, loc_name, lat, lon
            )
            VALUES (?, ?, ?, ?)
        """, (
            station["id"],
            station.get("name"),
            lat,
            lon
        ))
    conn.commit()


def save_rainfall_data(rows, conn, snapshot_time):
    cur = conn.cursor()
    readings = rows["data"]["readings"]
   

    # Insert rainfall data
    # Dynamic data, rows inserted in intervals with station id as foreign key
    for reading in readings:
        reading_time = reading["timestamp"]
        for entry in reading["data"]:

            station_id = entry["stationId"]
            value_mm = float(entry["value"])
            cur.execute("""
                INSERT OR IGNORE INTO rainfall (
                    station_id, value_mm, reading_time, snapshot_time
                )
                VALUES (?, ?, ?, ?)
            """, (
                station_id,
                value_mm,
                reading_time,
                snapshot_time
            ))

    conn.commit()


def main():
    conn = initialize_sqlite()
    print("Connected to trafficdata.db")
    # rainfall_data = get_rainfall_data()
    # save_rainfall_stations(rainfall_data, conn)

    sleep_until_next()

    try:
        while True:

            if TIME_BASED_POLLING == True:
                now_utc = datetime.datetime.now(datetime.timezone.utc)
                if not in_active_window(now_utc):
                    print(f"{now_utc} Outside active window, sleeping until 6am SGT")
                    sleep_until_next_window(now_utc)
                    continue

            # Record time for this cycle
            snapshot_time = datetime.datetime.now(datetime.timezone.utc).isoformat()

            try:
                # Save Speedbands data
                speedbands_data = get_speedbands_data()
                print(f"[{datetime.datetime.now()}] Get SpeedBands: {len(speedbands_data)} rows")
                save_speedbands_data(speedbands_data, conn, snapshot_time)
                print("Saved Speedbands data")
            except Exception as e:
                print("Error speedbands: ", e)

            try:
                # Save Incidents data
                incidents_data = get_incidents_data()
                print(f"[{datetime.datetime.now()}] Get Incidents: {len(incidents_data)} rows")
                save_incidents_data(incidents_data, conn, snapshot_time)
                print("Saved incidents data")
            except Exception as e:
                print("Error incidents: ", e)

            try:
                # Save estimated_tt data
                est_trav_time_data = get_estimated_tt_data()
                print(f"[{datetime.datetime.now()}] Get Estimated Travel Times: {len(est_trav_time_data)} rows")
                save_estimated_tt_data(est_trav_time_data, conn, snapshot_time)
                print("Saved Estimated Travel Times data")
            except Exception as e:
                print("Error estimated travel times: ", e)

            try:
                # Save Faulty Traffic Lights data
                faulty_traffic_lights_data = get_faulty_tl_data()
                print(f"[{datetime.datetime.now()}] Get Faulty Traffic Lights: {len(faulty_traffic_lights_data)} rows")
                save_faulty_tl_data(faulty_traffic_lights_data, conn, snapshot_time)
                print("Saved Faulty Traffic Lights data")
            except Exception as e:
                print("Error: ", e)

            try:
                # Save VMS data
                vms_data = get_vms_data()
                print(f"[{datetime.datetime.now()}] Get VMS: {len(vms_data)} rows")
                save_vms_data(vms_data, conn, snapshot_time)
                print("Saved VMS data")
            except Exception as e:
                print("Error VMS: ", e)

            try:
                # Save Train Service Alerts data
                train_service_alerts_data = get_train_service_alerts_data()
                print(f"[{datetime.datetime.now()}] Get Train Service Alerts: {len(train_service_alerts_data)} rows")
                save_train_service_alerts_data(train_service_alerts_data, conn, snapshot_time)
                print("Saved Train Service Alerts data")
            except Exception as e:
                print("Error Train Service alerts: ", e)

                # Save Rainfall data
                
            try:
                rainfall_data = get_rainfall_data()
                num_stations = len(rainfall_data["data"]["stations"])
                num_readings = len(rainfall_data["data"]["readings"])
                print(f"[{datetime.datetime.now()}] Get Rainfall: {num_stations} stations, {num_readings} entries")
                save_rainfall_data(rainfall_data, conn, snapshot_time)
                print("Saved Rainfall data")
            except Exception as e:
                print("Error Rainfall: ", e)

            

            sleep_until_next()
    finally:
        conn.close()
        print("Database connection closed")

if __name__ == "__main__":
    main()


