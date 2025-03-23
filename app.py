import os
import json
import threading
import time
from flask import Flask, render_template
import firebase_admin
from firebase_admin import credentials, db
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.cluster import DBSCAN
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Read Firebase Credentials
firebase_credentials_path = os.getenv("FIREBASE_CREDENTIALS_PATH")
firebase_db_url = os.getenv("FIREBASE_DB_URL")

with open(firebase_credentials_path, "r") as f:
    firebase_credentials = json.load(f)

# Initialize Firebase
cred = credentials.Certificate(firebase_credentials)
firebase_admin.initialize_app(cred, {'databaseURL': firebase_db_url})

# Flask Server
app = Flask(__name__)

def fetch_data():
    """Fetch data from Firebase Realtime Database."""
    ref = db.reference('data')
    data = ref.get()
    records = []

    if not data:
        print("No data found in Firebase.")
        return pd.DataFrame(columns=['latitude', 'longitude', 'timestamp'])

    for key, value in data.items():
        if 'latitude' in value and 'longitude' in value and 'time' in value:
            try:
                records.append({
                    'latitude': float(value['latitude']),
                    'longitude': float(value['longitude']),
                    'timestamp': value['time']
                })
            except ValueError:
                print(f"Skipping invalid record: {value}")  # Debugging
        else:
            print(f"Skipping incomplete record: {value}")  # Debugging

    return pd.DataFrame(records)

def parse_timestamp(timestamp):
    """Parse timestamp from Unix epoch (seconds/milliseconds) or formatted string."""
    if isinstance(timestamp, (int, float)):
        try:
            if timestamp > 10**10:  # Likely in milliseconds
                timestamp /= 1000  # Convert to seconds
            dt = datetime.utcfromtimestamp(timestamp)
            return pd.Series({'hour': dt.hour, 'day_of_week': dt.weekday(), 'weekend': 1 if dt.weekday() >= 5 else 0})
        except (ValueError, OSError) as e:
            print(f"Epoch timestamp error: {timestamp} -> {e}")

    elif isinstance(timestamp, str):
        try:
            dt = datetime.strptime(timestamp, "%a %b %d %H:%M:%S GMT%z %Y")
            return pd.Series({'hour': dt.hour, 'day_of_week': dt.weekday(), 'weekend': 1 if dt.weekday() >= 5 else 0})
        except ValueError as e:
            print(f"String timestamp parsing error: {timestamp} -> {e}")

    return pd.Series({'hour': None, 'day_of_week': None, 'weekend': None})

def process_time(df):
    """Process time-related features from timestamps."""
    time_features = df['timestamp'].apply(parse_timestamp)
    df = pd.concat([df, time_features], axis=1)
    return df.drop(columns=['timestamp'])

def crime_zone_clustering(df):
    """Apply DBSCAN clustering to classify crime zones."""
    if df.empty:
        print("No valid data for clustering.")
        df['crime_zone'] = np.nan
        df['zone_category'] = 'Low'
        return df

    coords = df[['latitude', 'longitude']].dropna().values  # Drop NaN values

    if len(coords) < 3:
        print("Not enough data points for clustering.")
        df['crime_zone'] = np.nan
        df['zone_category'] = 'Low'
        return df

    clustering = DBSCAN(eps=0.0005, min_samples=3, metric='haversine').fit(np.radians(coords))
    df['crime_zone'] = clustering.labels_
    df['zone_category'] = df['crime_zone'].apply(lambda x: 'High' if x != -1 else 'Low')

    return df

def run_ml_model():
    """Periodically fetch, process, and analyze crime data."""
    while True:
        print("Running ML Model...")
        df = fetch_data()

        if df.empty:
            print("No data available. Skipping processing.")
        else:
            df = process_time(df)
            df = crime_zone_clustering(df)
            df.to_csv("crime_zone_output.csv", index=False)
            print("Output saved.")

        time.sleep(3600)  # Run every hour

# Start ML Model in Background
threading.Thread(target=run_ml_model, daemon=True).start()

@app.route("/")
def home():
    """Render the HTML page with the latest crime zone data."""
    try:
        df = pd.read_csv("crime_zone_output.csv")
    except FileNotFoundError:
        df = pd.DataFrame(columns=['latitude', 'longitude', 'crime_zone', 'zone_category'])

    return render_template("index.html", tables=[df.to_html(classes='data')], titles=df.columns.values)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
