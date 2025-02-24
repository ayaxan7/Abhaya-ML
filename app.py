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

# Load Firebase Credentials from .env
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

# Read the JSON file
firebase_credentials_path = os.getenv("FIREBASE_CREDENTIALS_PATH")
with open(firebase_credentials_path, "r") as f:
    firebase_credentials = json.load(f)

print(firebase_credentials["project_id"])  # Verify
# Initialize Firebase
cred = credentials.Certificate(firebase_credentials)
firebase_admin.initialize_app(cred, {
    'databaseURL': os.getenv("FIREBASE_DB_URL")
})
# Flask App
app = Flask(__name__)

def fetch_data():
    ref = db.reference('data')
    data = ref.get()
    records = []

    for key, value in data.items():
        records.append({
            'latitude': float(value['latitude']),
            'longitude': float(value['longitude']),
            'timestamp': value['time']
        })

    return pd.DataFrame(records)

def process_time(df):
    def parse_timestamp(timestamp_str):
        if not isinstance(timestamp_str, str):
            timestamp_str = str(timestamp_str)

        try:
            dt = datetime.strptime(timestamp_str, "%a %b %d %H:%M:%S GMT%z %Y")
        except ValueError:
            return pd.Series({'hour': None, 'day_of_week': None, 'weekend': None})
        
        return pd.Series({
            'hour': dt.hour,
            'day_of_week': dt.weekday(),
            'weekend': 1 if dt.weekday() >= 5 else 0
        })

    time_features = df['timestamp'].apply(parse_timestamp)
    df = pd.concat([df, time_features], axis=1)
    return df.drop(columns=['timestamp'])

def crime_zone_clustering(df):
    coords = df[['latitude', 'longitude']].values
    clustering = DBSCAN(eps=0.001, min_samples=3, metric='haversine').fit(np.radians(coords))
    df['crime_zone'] = clustering.labels_
    
    df['zone_category'] = df['crime_zone'].apply(lambda x: 'High' if x != -1 else 'Low')
    return df

def run_ml_model():
    while True:
        print("Running ML Model...")
        df = fetch_data()
        df = process_time(df)
        df = crime_zone_clustering(df)
        df.to_csv("crime_zone_output.csv", index=False)
        print("Output saved.")

        # Sleep for 1 hour
        time.sleep(3600)

# Start ML Model in Background
threading.Thread(target=run_ml_model, daemon=True).start()

@app.route("/")
def home():
    df = pd.read_csv("crime_zone_output.csv")
    return render_template("index.html", tables=[df.to_html()], titles=df.columns.values)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
