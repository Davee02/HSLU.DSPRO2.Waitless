#!/usr/bin/env python3
"""
Queue Times Scraper Module

This script logs into queue-times.com, navigates through a date range,
extracts ride wait times and weather data from Chartkick charts, and saves
the collected data into Parquet checkpoint files regularly to avoid excessive
memory usage.

The script now accepts a park ID parameter. By default it uses Europa Park (ID 51),
but you can run it for Rulantica (ID 309) via the command-line argument --park-id.
"""

import time
import os
import argparse
import traceback
from datetime import date, datetime, timedelta
from collections import defaultdict
from random import random
from typing import List

import pandas as pd
from playwright.sync_api import sync_playwright

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import time

# Initialize Firebase Admin SDK if not already initialized
if not firebase_admin._apps:
    # You'll need to provide your Firebase Admin SDK credentials file
    # This file should be downloaded from your Firebase project settings
    # and its path should be accessible to the script.
    # For local development, you can set the GOOGLE_APPLICATION_CREDENTIALS
    # environment variable to the path of your credentials file.
    try:
        cred = credentials.ApplicationDefault()
        firebase_admin.initialize_app(cred)
    except Exception as e:
        print(f"Error initializing Firebase Admin SDK: {e}")
        print("Please ensure GOOGLE_APPLICATION_CREDENTIALS environment variable is set correctly.")
        exit(1)


def get_date_range(start_date: date, end_date: date) -> List[date]:
    """Generate a list of dates from start_date to end_date (inclusive)."""
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date)
        current_date += timedelta(days=1)
    return dates


class QueueTimesScraper:
    """A scraper for extracting ride wait times and weather data from queue-times.com."""
    
    LOGIN_URL = "https://queue-times.com/users/sign_in"
    
    def __init__(self, page, checkpoint_save_dir, park_id: int, verbose: bool = False) -> None:
        """
        Initialize the scraper with a Playwright page, checkpoint directory,
        park id, and empty data lists.
        
        Args:
            page: The Playwright page instance.
            checkpoint_save_dir: Directory to save checkpoint files.
            park_id: The park ID to scrape (e.g., 51 for Europa Park, 309 for Rulantica).
            verbose: If True, enable verbose logging.
        """
        self.page = page
        self.checkpoint_save_dir = checkpoint_save_dir
        self.ride_data: List[dict] = []
        self.weather_data: List[dict] = []
        self.verbose = verbose
        self.park_id = park_id
        # Create a URL template using the given park id.
        self.CALENDAR_URL_TEMPLATE = f"https://queue-times.com/parks/{park_id}/calendar/{{year}}/{{month:02d}}/{{day:02d}}"
        self.db = firestore.client()

    def trace(self, message: str) -> None:
        if self.verbose:
            print(message)

    def log(self, message: str) -> None:
        print(message)

    def error(self, message: str) -> None:
        print(message)

    def login(self, username: str, password: str) -> None:
        """
        Perform login with the provided credentials, including cookie handling.\n        \n        Args:\n            username: The user email.\n            password: The user password.\n        """
        try:
            self.log("Navigating to the login page")
            self.page.goto(self.LOGIN_URL)
            
            # Handle the cookie banner
            self.page.wait_for_selector("a.cmpboxbtnyes", timeout=5000)
            self.page.click("a.cmpboxbtnyes")
            self.page.wait_for_selector("div#cmpbox", state="hidden", timeout=5000)
            
            # Fill in login details
            self.page.fill("#user_email", username)
            self.page.fill("#user_password", password)
            self.page.check("#user_remember_me")
            self.page.click("input[type='submit']")
        except Exception as e:
            self.error("Error during login: " + str(e))
            raise

    def go_to_date(self, current_date: date) -> None:
        """
        Navigate to the calendar page for the specified date.
        
        Args:
            current_date: The date for which to load the calendar page.
        """
        try:
            url = self.CALENDAR_URL_TEMPLATE.format(
                year=current_date.year,
                month=current_date.month,
                day=current_date.day
            )
            self.page.goto(url, wait_until="domcontentloaded")
        except Exception as e:
            self.error(f"Error navigating to {current_date}: {e}")
            raise

    def process_date(self, current_date: date) -> None:
        """
        Process a single date: extract ride wait times and weather data,
        then store the data in internal lists and save to Firestore.
        
        Args:
            current_date: The date to be processed.
        """
        self.log(f"Processing data for {current_date}")
        # Clear data from previous date
        self.ride_data.clear()
        self.weather_data.clear()

        try:
            self.go_to_date(current_date)
            time.sleep(1 + random())

            page_content = self.page.inner_text("body")
            if "The page you were looking for doesn't exist" in page_content:
                self.log("No data available for this date")
                return

            all_charts = self.page.evaluate('Chartkick.charts')
            self.trace(f"Found {len(all_charts)} Chartkick charts on the page")

            # Process ride charts (assumed to be indexed from 7 onward)
            for i in range(7, len(all_charts)):
                chart_id = f"chart-{i}"
                self.trace(f"Getting data for chart {chart_id}")

                # Extract ride name from the page DOM via JavaScript evaluation
                ride_name_js = (
                    f'Chartkick.charts["{chart_id}"].element.parentElement.'
                    'previousElementSibling.children[0].children[0].text'
                )
                try:
                    ride_name = self.page.evaluate(ride_name_js)
                    self.trace(f"Ride name: {ride_name}")
                except Exception as e:
                    self.error(f"Error extracting ride name for chart {chart_id}: {e}")
                    continue

                wait_times_js = f'Chartkick.charts["{chart_id}"].rawData[1].data'
                try:
                    wait_times = self.page.evaluate(wait_times_js)
                    self.trace(f"Wait times: {wait_times}")
                except Exception as e:
                    self.error(f"Error extracting wait times for chart {chart_id}: {e}")
                    continue

                # For each timestamp/wait time pair, parse and store the data
                for time_str, wait in wait_times:
                    try:
                        ride_timestamp = datetime.strptime(time_str, "%m/%d/%y %H:%M:%S")
                    except ValueError:
                        self.error(f"Timestamp format error for ride: {time_str}")
                        continue
                    record = {
                        "timestamp": ride_timestamp,
                        "ride_name": ride_name,
                        "wait_time": float(wait)
                    }
                    self.ride_data.append(record)
                self.trace("")


            # Process weather data (temperature, rain, wind)
            try:
                temperature_data = self.page.evaluate('Chartkick.charts["chart-4"].rawData[0].data')
                rain_data = self.page.evaluate('Chartkick.charts["chart-5"].rawData[0].data')
                wind_data = self.page.evaluate('Chartkick.charts["chart-6"].rawData[0].data')
            except Exception as e:
                self.error(f"Error extracting weather data for {current_date}: {e}")
                # Continue to the next date even if weather data extraction fails
                return

            self.trace(f"Temperature data: {temperature_data}")
            self.trace(f"Rain data: {rain_data}")
            self.trace(f"Wind data: {wind_data}")

            # Assuming all three weather datasets have the same timestamps:
            if temperature_data and rain_data and wind_data:
                for (time_str, temperature), (_, rain), (_, wind) in zip(temperature_data, rain_data, wind_data):
                    try:
                        weather_timestamp = datetime.strptime(time_str, "%m/%d/%y %H:%M:%S")
                    except ValueError:
                        self.error(f"Timestamp format error for weather: {time_str}")
                        continue
                    record = {
                        "timestamp": weather_timestamp,
                        "temperature": temperature,
                        "rain": rain,
                        "wind": wind
                    }
                    self.weather_data.append(record)
                # TODO: Add logic to save weather data to Firestore
                self.trace(f"Processed weather data for {current_date}\n")
        except Exception as e:
            self.error(f"Error processing date {current_date}: {e}")
            self.error(traceback.format_exc())

    def save_checkpoint(self, checkpoint_index: int) -> None:
        """
        Save the current ride and weather data to Parquet files as a checkpoint
        and clear the in-memory data to free up memory. This is now mainly for backup
        or further processing outside of Firestore.
        
        Args:
            checkpoint_index: An index used in naming the checkpoint files.
        """
        try:
            if self.ride_data:
                rides_df = pd.DataFrame(self.ride_data)
                rides_file = f"ride_queue_times_checkpoint_{checkpoint_index}.parquet"
                rides_file = os.path.join(self.checkpoint_save_dir, rides_file)
                rides_df.to_parquet(rides_file, engine="pyarrow")
                self.log(f"Checkpoint {checkpoint_index}: Saved {len(rides_df)} ride records to '{rides_file}'")
            else:
                self.log(f"Checkpoint {checkpoint_index}: No ride data to save to checkpoint.")

            if self.weather_data:
                weather_df = pd.DataFrame(self.weather_data)
                weather_file = f"weather_data_checkpoint_{checkpoint_index}.parquet"
                weather_file = os.path.join(self.checkpoint_save_dir, weather_file)
                weather_df.to_parquet(weather_file, engine="pyarrow")
                self.log(f"Checkpoint {checkpoint_index}: Saved {len(weather_df)} weather records to '{weather_file}'")
            else:
                self.log(f"Checkpoint {checkpoint_index}: No weather data to save to checkpoint.")

            # Clear in-memory data is now done at the beginning of process_date
            # self.ride_data.clear()
            # self.weather_data.clear()
        except Exception as e:
            self.error(f"Error saving checkpoint {checkpoint_index}: {e}")
            self.error(traceback.format_exc())
            
    def save_all_ride_data_to_firestore(self) -> None:
        """
        Saves all accumulated ride data (self.ride_data) to Firestore using batched writes.
        Data is grouped by ride_name, and each entry is added to the 'queueTimes'
        subcollection of the corresponding attraction document.
        """
        self.log(f"Saving all accumulated ride data to Firestore ({len(self.ride_data)} records)")
        if not self.ride_data:
            self.log("No ride data to save to Firestore.")
            return

        # Group ride data by ride name
        ride_data_grouped = defaultdict(list)
        for record in self.ride_data:
            ride_data_grouped[record["ride_name"]].append(record)

        total_records_saved = 0
        batch = self.db.batch()
        batch_size = 0
        max_batch_size = 500 # Maximum operations per batch in Firestore

        for ride_name, records in ride_data_grouped.items():
            attraction_ref = self.db.collection('attractions').document(ride_name)
            for record in records:
                # Prepare the data for the subcollection document
                record_data = {
                    "timestamp": record["timestamp"],
                    "wait_time": float(record["wait_time"]) # Ensure wait_time is float
                }
                # Add a set operation to the batch for the subcollection document
                batch.set(attraction_ref.collection('queueTimes').document(), record_data)
                batch_size += 1
                total_records_saved += 1

                if batch_size >= max_batch_size:
                    try:
                        batch.commit()
                        self.log(f"Committed batch with {batch_size} writes. Total saved: {total_records_saved}")
                        batch = self.db.batch()
                        batch_size = 0
                        time.sleep(60) # Add this line
                    except Exception as e:
                        self.error(f"Error committing batch: {e}")
                        # Depending on your error handling needs, you might want to
                        # implement retry logic or stop the process here.
                        raise

        # Commit any remaining writes in the last batch
        if batch_size > 0:
            try:
                batch.commit()
                self.log(f"Committed final batch with {batch_size} writes. Total saved: {total_records_saved}")
                time.sleep(60) # Add this line
            except Exception as e:
                self.error(f"Error committing final batch: {e}")
                raise

        self.log(f"Finished saving all {total_records_saved} ride records to Firestore.")

def main() -> None:
    """Main function to run the scraper with periodic checkpointing."""
    parser = argparse.ArgumentParser(description="Queue Times Scraper")
    parser.add_argument("--park-id", type=int, default=51,
                        help="ID of the amusement park to scrape (default: 51 for Europa Park; use 309 for Rulantica)")
    parser.add_argument("--login-email", type=str, help="Email address for login on queue-times.com")
    parser.add_argument("--login-password", type=str, help="Password for login on queue-times.com")
    parser.add_argument("--start-date", type=str, default="2024-01-01", help="Start date for scraping in YYYY-MM-DD format")
    parser.add_argument("--end-date", type=str, default="2024-12-31", help="End date for scraping in YYYY-MM-DD format")
    parser.add_argument("--skip-weather", action="store_true", help="Skip weather data extraction")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    # Define the date range
    verbose = args.verbose

    start_date = date.fromisoformat(args.start_date)
    end_date = date.fromisoformat(args.end_date)
    date_range = get_date_range(start_date, end_date)

    checkpoint_interval = 30 # Save checkpoint every 5 days processed.
    checkpoint_index = 1
    checkpoint_save_dir = os.path.join("../data/raw/checkpoints", str(args.park_id))

    os.makedirs(checkpoint_save_dir, exist_ok=True)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            context = browser.new_context(locale="en-US")

            if verbose:
                print("Opening new page")
            page = context.new_page()

            scraper = QueueTimesScraper(page, checkpoint_save_dir, park_id=args.park_id, verbose=verbose)
            scraper.login(args.login_email, args.login_password)

            for idx, current_date in enumerate(date_range, start=1):
                try:
                    scraper.process_date(current_date)
                except Exception as e:
                    scraper.error(f"Unhandled error processing date {current_date}: {e}")
                    scraper.error(traceback.format_exc())
                # Save a checkpoint every `checkpoint_interval` days.
                # This is now mainly for backup, as data is saved to Firestore per date.
                if idx % checkpoint_interval == 0:
                    try:
                        scraper.save_checkpoint(checkpoint_index)
                        checkpoint_index += 1
                    except Exception as e:
                        scraper.error(f"Unhandled error saving checkpoint {checkpoint_index}: {e}")
                        scraper.error(traceback.format_exc())

            # Save all accumulated ride data to Firestore after processing all dates
            if scraper.ride_data:
                try:
                    scraper.save_all_ride_data_to_firestore()
                except Exception as e:
                    scraper.error(f"Unhandled error saving data to Firestore: {e}")
                    scraper.error(traceback.format_exc())

            context.close()
            browser.close()
    except Exception as e:
        print(f"Critical error in main: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()