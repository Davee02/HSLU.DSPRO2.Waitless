#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to gather historical weather data for Rust, Germany from Open-Meteo API,
keeping only rain_mm and temperature_C, processing into 5-minute bins through
interpolation, and export as parquet.
"""

import os
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import firebase_admin
from firebase_admin import credentials, firestore
import pyarrow as pa
import pyarrow.parquet as pq

OUTPUT_DIR = "data/raw/weather"
RUST_LAT = 48.2664  # Latitude for Rust, Germany
RUST_LON = 7.7224   # Longitude for Rust, Germany
START_YEAR = 2017
END_YEAR = datetime.now().year 
API_BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Initialize Firebase Admin SDK
if not firebase_admin._apps:
    cred = credentials.ApplicationDefault()
    firebase_admin.initialize_app(cred)
WEATHER_LOCATION_ID = 'rust_germany'


os.makedirs(OUTPUT_DIR, exist_ok=True)

def fetch_open_meteo_data(lat, lon, start_date, end_date):
    """
    Fetch weather data from Open-Meteo API for the given coordinates and date range.
    Only retrieve temperature and rain data.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,rain",
        "timezone": "Europe/Berlin"
    }
    
    url = f"{API_BASE_URL}"
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  
        
        data = response.json()
        
        # Extract hourly data - only temperature and rain
        hourly_data = {
            "timestamp": pd.to_datetime(data["hourly"]["time"]),
            "temperature_C": data["hourly"]["temperature_2m"],
            "rain_mm": data["hourly"]["rain"]
        }
        
        df = pd.DataFrame(hourly_data)
        return df
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Open-Meteo API: {e}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            print(f"Response: {e.response.text}")
        return None

def interpolate_to_5min(hourly_df):
    """
    Interpolate hourly data to 5-minute intervals.
    Only process temperature and rain columns.
    """
    if hourly_df is None or hourly_df.empty:
        return None
    
    start_time = hourly_df['timestamp'].min()
    end_time = hourly_df['timestamp'].max()
    time_range_5min = pd.date_range(start=start_time, end=end_time, freq='5min')
    
    df_5min = pd.DataFrame({'timestamp': time_range_5min})
    
    df_merged = pd.merge(df_5min, hourly_df, on='timestamp', how='left')
    
    # Interpolate temperature using cubic interpolation
    df_merged['temperature_C'] = df_merged['temperature_C'].interpolate(method='cubic')
    
    # For rain, distribute hourly values evenly across the hour
    if 'rain_mm' in df_merged.columns:

        df_merged['rain_hourly'] = df_merged['rain_mm'].ffill()
        intervals_per_hour = 60 // 5
        df_merged['rain_mm_per_hour'] = df_merged['rain_hourly']
        df_merged.drop('rain_hourly', axis=1, inplace=True)
    
    return df_merged

def process_data_by_year(lat, lon, year):
    """
    Process data for a specific year, retrieving only temperature and rain.
    """
    print(f"Processing data for year {year}...")
    
    start_date = f"{year}-01-01"
    
    if year == datetime.now().year:
        end_date = datetime.now().strftime("%Y-%m-%d")
    else:
        end_date = f"{year}-12-31"
    

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    chunk_dfs = []
    current_start = start_dt
    
    while current_start <= end_dt:

        current_end = min(current_start + timedelta(days=90), end_dt)
        
        chunk_start = current_start.strftime("%Y-%m-%d")
        chunk_end = current_end.strftime("%Y-%m-%d")
        
        print(f"  Fetching data from {chunk_start} to {chunk_end}...")

        chunk_df = fetch_open_meteo_data(lat, lon, chunk_start, chunk_end)
        
        if chunk_df is not None and not chunk_df.empty:
            chunk_dfs.append(chunk_df)
        else:
            print(f"  Failed to fetch data for {chunk_start} to {chunk_end}")
        
        current_start = current_end + timedelta(days=1)

        time.sleep(1)
    
    if chunk_dfs:
        hourly_df = pd.concat(chunk_dfs, ignore_index=True)
        hourly_df = hourly_df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')

        df_5min = interpolate_to_5min(hourly_df)
        return df_5min
    else:
        print(f"  No data fetched for year {year}")
        return None

def save_to_parquet(df, output_file='rust_weather_5min_rain_temperature.parquet'):
    """
    Save the DataFrame to a Parquet file.
    """
    full_path = os.path.join(OUTPUT_DIR, output_file)
    
    table = pa.Table.from_pandas(df)
    pq.write_table(table, full_path, compression='snappy')
    
    print(f"Data saved to {full_path}")
    return full_path

def main():
    def save_weather_data_to_firestore(df):
        """
        Save the weather DataFrame to Firestore.
        """
        db = firestore.client()
        weather_location_ref = db.collection('weatherLocations').document(WEATHER_LOCATION_ID)

        print(f"Saving {len(df)} weather records to Firestore...")
        batch = db.batch()
        for index, row in df.iterrows():
            reading_data = {
                'timestamp': row['timestamp'],
                'temperature_C': row['temperature_C'],
                'rain_mm': row.get('rain_mm_per_hour', row.get('rain_mm')) # Use interpolated rain if available
            }
            batch.set(weather_location_ref.collection('readings').document(), reading_data)
        batch.commit()
    """
    Main function to execute the data collection and processing.
    Only collect temperature and rain data.
    """
    print(f"Starting historical weather data collection for Rust, Germany ({START_YEAR}-{END_YEAR})...")
    print("Collecting only temperature_C and rain_mm data.")
    
    all_dfs = []
    
    for year in range(START_YEAR, END_YEAR + 1):
        df = process_data_by_year(RUST_LAT, RUST_LON, year)
        
        if df is not None and not df.empty:
            all_dfs.append(df)
    
    if all_dfs:
        print("Combining all years of data...")
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')

        print("\nBasic Statistics:")
        print(combined_df.describe())

        missing_values = combined_df.isna().sum()
        print("\nMissing values per column:")
        print(missing_values)

        # Save data to Firestore
        save_weather_data_to_firestore(combined_df)

        output_file = save_to_parquet(combined_df)

        print("\nLoading the parquet file to verify...")
        loaded_df = pd.read_parquet(output_file)
        print(f"Loaded dataframe shape: {loaded_df.shape}")
        print("First few rows:")
        print(loaded_df.head())
        
        return loaded_df
    else:
        print("No data collected. Please check errors above.")
        return None

if __name__ == "__main__":
    df = main()