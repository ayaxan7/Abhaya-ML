import firebase_admin
from firebase_admin import credentials, db
import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.cluster import DBSCAN

# Initialize Firebase
cred = credentials.Certificate("deleted-project-6d75e-firebase-adminsdk-avwl1-78b2ae6fd1.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://deleted-project-6d75e-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

# Fetch Data from Firebase
def fetch_data():
    ref = db.reference('/data')  
    data = ref.get()
    records = []

    for key, value in data.items():
        records.append({
            'latitude': float(value['latitude']),
            'longitude': float(value['longitude']),
            'timestamp': value['time']
        })

    return pd.DataFrame(records)

# Convert Timestamp to Features
def process_time(df):
    def parse_timestamp(timestamp_str):
        # Ensure timestamp is a string
        if not isinstance(timestamp_str, str):
            timestamp_str = str(timestamp_str)

        try:
            dt = datetime.strptime(timestamp_str, "%a %b %d %H:%M:%S GMT%z %Y")
        except ValueError:
            return pd.Series({'hour': None, 'day_of_week': None, 'weekend': None})  # Handle invalid timestamps
        
        return pd.Series({
            'hour': dt.hour,
            'day_of_week': dt.weekday(),
            'weekend': 1 if dt.weekday() >= 5 else 0
        })

    time_features = df['timestamp'].apply(parse_timestamp)
    df = pd.concat([df, time_features], axis=1)
    return df.drop(columns=['timestamp'])

# Crime Zone Clustering using DBSCAN
def crime_zone_clustering(df):
    coords = df[['latitude', 'longitude']].values
    clustering = DBSCAN(eps=0.001, min_samples=3, metric='haversine').fit(np.radians(coords))
    df['crime_zone'] = clustering.labels_
    
    # Map cluster labels to categories
    df['zone_category'] = df['crime_zone'].apply(lambda x: 'High' if x != -1 else 'Low')
    return df

# Real-Time Execution
def main():
    df = fetch_data()
    df = process_time(df)
    df = crime_zone_clustering(df)

    # Save to CSV
    df.to_csv("crime_zone_output.csv", index=False)

    print("Output saved to crime_zone_output.csv")
    print(df[['latitude', 'longitude', 'zone_category']])  # Display categorized zones

if __name__ == "__main__":
    main()
