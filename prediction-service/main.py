from flask import Flask, request, jsonify
import os
import logging
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import requests
import holidays
from google.cloud import firestore, storage
from google.cloud.firestore_v1 import FieldFilter
import threading
import gc
from functools import lru_cache

# Import your model classes
from ml_models.tcn_model import AutoregressiveTCNModel
from inference.predictor import WaitTimePredictor

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model caching
model_cache = {}
cache_lock = threading.Lock()
MAX_MODELS_IN_MEMORY = 3  # Adjust based on Cloud Run memory limits

# Initialize Firestore
db = firestore.Client()

# Initialize Cloud Storage for model files
storage_client = storage.Client()
MODEL_BUCKET = os.environ.get('MODEL_BUCKET', 'waitless-aca8b-prediction-models')

# Europa-Park coordinates (Rust, Germany)
RUST_LATITUDE = 48.266
RUST_LONGITUDE = 7.722

# Ride mapping (from your queueTimesService.ts)
LOCAL_RIDE_MAPPINGS = [
    {"id": "silver-star", "name": "Silver Star", "normalized_name": "Silver_Star"},
    {"id": "blue-fire", "name": "Blue Fire Megacoaster", "normalized_name": "Blue_Fire_Megacoaster"},
    {"id": "wodan", "name": "Wodan – Timburcoaster", "normalized_name": "Wodan_Timburcoaster"},
    {"id": "voletarium", "name": "Voletarium", "normalized_name": "Voletarium"},
    {"id": "alpine-express-enzian", "name": "Alpine Express Enzian", "normalized_name": "Alpine_Express_Enzian"},
    {"id": "arena-of-football", "name": "Arena of Football - Be Part of It", "normalized_name": "Arena_of_Football_Be_Part_of_It"},
    {"id": "arthur", "name": "Arthur", "normalized_name": "Arthur"},
    {"id": "atlantica-supersplash", "name": "Atlantica SuperSplash", "normalized_name": "Atlantica_SuperSplash"},
    {"id": "atlantis-adventure", "name": "Atlantis Adventure", "normalized_name": "Atlantis_Adventure"},
    {"id": "baaa-express", "name": "Baaa Express", "normalized_name": "Baaa_Express"},
    {"id": "bellevue-ferris-wheel", "name": "Bellevue Ferris Wheel", "normalized_name": "Bellevue_Ferris_Wheel"},
    {"id": "castello-dei-medici", "name": "Castello dei Medici", "normalized_name": "Castello_dei_Medici"},
    {"id": "dancing-dingie", "name": "Dancing Dingie", "normalized_name": "Dancing_Dingie"},
    {"id": "euromir", "name": "Euro-Mir", "normalized_name": "Euro_Mir"},
    {"id": "eurosat-cancan-coaster", "name": "Eurosat CanCan Coaster", "normalized_name": "Eurosat_CanCan_Coaster"},
    {"id": "eurotower", "name": "Euro-Tower", "normalized_name": "Euro_Tower"},
    {"id": "fjordrafting", "name": "Fjord-Rafting", "normalized_name": "Fjord_Rafting"},
    {"id": "jim-button-journey", "name": "Jim Button - Journey Through Morrowland", "normalized_name": "Jim_Button_Journey_Through_Morrowland"},
    {"id": "josefinas-imperial-journey", "name": "Josefina's Magical Imperial Journey", "normalized_name": "Josefinas_Magical_Imperial_Journey"},
    {"id": "kolumbusjolle", "name": "Kolumbusjolle", "normalized_name": "Kolumbusjolle"},
    {"id": "madame-freudenreich", "name": "Madame Freudenreich Curiosités", "normalized_name": "Madame_Freudenreich_Curiosites"},
    {"id": "matterhorn-blitz", "name": "Matterhorn-Blitz", "normalized_name": "Matterhorn_Blitz"},
    {"id": "old-macdonald", "name": "Old Mac Donald's Tractor Fun", "normalized_name": "Old_Mac_Donalds_Tractor_Fun"},
    {"id": "pegasus", "name": "Pegasus", "normalized_name": "Pegasus"},
    {"id": "pirates-in-batavia", "name": "Pirates in Batavia", "normalized_name": "Pirates_in_Batavia"},
    {"id": "poppy-towers", "name": "Poppy Towers", "normalized_name": "Poppy_Towers"},
    {"id": "poseidon", "name": "Poseidon", "normalized_name": "Poseidon"},
    {"id": "snorri-touren", "name": "Snorri Touren", "normalized_name": "Snorri_Touren"},
    {"id": "swiss-bob-run", "name": "Swiss Bob Run", "normalized_name": "Swiss_Bob_Run"},
    {"id": "tirol-log-flume", "name": "Tirol Log Flume", "normalized_name": "Tirol_Log_Flume"},
    {"id": "tnnevirvel", "name": "Tnnevirvel", "normalized_name": "Tnnevirvel"},
    {"id": "vienna-wave-swing", "name": "Vienna Wave Swing - Glckspilz", "normalized_name": "Vienna_Wave_Swing_Glckspilz"},
    {"id": "vindjammer", "name": "Vindjammer", "normalized_name": "Vindjammer"},
    {"id": "volo-da-vinci", "name": "Volo da Vinci", "normalized_name": "Volo_da_Vinci"},
    {"id": "voltron-nevera", "name": "Voltron Nevera - Powered by Rimac", "normalized_name": "Voltron_Nevera_Powered_by_Rimac"},
    {"id": "whale-adventures", "name": "Whale Adventures - Northern Lights", "normalized_name": "Whale_Adventures_Northern_Lights"}
]

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

class ModelManager:
    """Manages model loading and caching"""
    
    def __init__(self):
        self.bucket = storage_client.bucket(MODEL_BUCKET)
        
    def get_model(self, ride_name: str) -> WaitTimePredictor:
        """Get model from cache or load from storage"""
        with cache_lock:
            if ride_name in model_cache:
                logger.info(f"Using cached model for {ride_name}")
                return model_cache[ride_name]
            
            # Check if we need to evict models to free memory
            if len(model_cache) >= MAX_MODELS_IN_MEMORY:
                self._evict_oldest_model()
            
            # Load model from Cloud Storage
            model = self._load_model_from_storage(ride_name)
            model_cache[ride_name] = model
            
            logger.info(f"Loaded and cached model for {ride_name}")
            return model
    
    def _load_model_from_storage(self, ride_name: str) -> WaitTimePredictor:
        """Load model files from Cloud Storage"""
    
        # Find the correct normalized name from LOCAL_RIDE_MAPPINGS
        normalized_name = None
        for mapping in LOCAL_RIDE_MAPPINGS:
            if mapping["name"] == ride_name:
                # CONVERT TO LOWERCASE - this was the missing piece!
                normalized_name = mapping["normalized_name"].lower()
                break
    
        if not normalized_name:
            # Fallback to simple conversion if not found in mappings
            normalized_name = ride_name.lower().replace(' ', '_').replace('-', '_')
            # Handle special characters that might be in ride names
            normalized_name = normalized_name.replace('–', '_').replace('(', '').replace(')', '').replace(',', '').replace('.', '').replace("'", '').replace('"', '')
            # Replace multiple underscores with single underscore
            import re
            normalized_name = re.sub(r'_+', '_', normalized_name)
            # Remove leading/trailing underscores
            normalized_name = normalized_name.strip('_')
    
        logger.info(f"Loading model for '{ride_name}' using normalized name: '{normalized_name}'")
    
        # Download model files to temp directory
        temp_dir = f"/tmp/{normalized_name}"
        os.makedirs(temp_dir, exist_ok=True)
    
        # Download GB model (with LOWERCASE naming)
        gb_blob = self.bucket.blob(f"models/{normalized_name}_gb_baseline.pkl")
        gb_path = f"{temp_dir}/{normalized_name}_gb_baseline.pkl"
    
        logger.info(f"Downloading GB model from: models/{normalized_name}_gb_baseline.pkl")
    
        try:
            gb_blob.download_to_filename(gb_path)
            logger.info(f"✅ Successfully downloaded GB model to {gb_path}")
        except Exception as e:
            logger.error(f"❌ Failed to download GB model: {e}")
            # List available models for debugging
            try:
                blobs = list(self.bucket.list_blobs(prefix="models/"))
                available_gb_models = [blob.name for blob in blobs if blob.name.endswith('_gb_baseline.pkl')]
                logger.error(f"Available GB models: {available_gb_models}")
            
                # Check for similar names
                similar_models = [name for name in available_gb_models if normalized_name in name.lower()]
                if similar_models:
                    logger.error(f"Similar model names found: {similar_models}")
            except Exception as list_e:
                logger.error(f"Could not list available models: {list_e}")
            raise
    
        # Download TCN model (with LOWERCASE naming) 
        tcn_blob = self.bucket.blob(f"models/{normalized_name}_cached_scheduled_sampling_tcn.pt")
        tcn_path = f"{temp_dir}/{normalized_name}_cached_scheduled_sampling_tcn.pt"
    
        logger.info(f"Downloading TCN model from: models/{normalized_name}_cached_scheduled_sampling_tcn.pt")
    
        try:
            tcn_blob.download_to_filename(tcn_path)
            logger.info(f"✅ Successfully downloaded TCN model to {tcn_path}")
        except Exception as e:
            logger.error(f"❌ Failed to download TCN model: {e}")
            # List available models for debugging
            try:
                blobs = list(self.bucket.list_blobs(prefix="models/"))
                available_tcn_models = [blob.name for blob in blobs if blob.name.endswith('_cached_scheduled_sampling_tcn.pt')]
                logger.error(f"Available TCN models: {available_tcn_models}")
            
                # Check for similar names
                similar_models = [name for name in available_tcn_models if normalized_name in name.lower()]
                if similar_models:
                    logger.error(f"Similar model names found: {similar_models}")
            except Exception as list_e:
                logger.error(f"Could not list available models: {list_e}")
            raise
    
        # Initialize predictor
        predictor = WaitTimePredictor(ride_name, temp_dir)
    
        logger.info(f"Successfully loaded models for '{ride_name}'")
        return predictor
    
    def _evict_oldest_model(self):
        """Remove oldest model from cache to free memory"""
        if model_cache:
            oldest_key = next(iter(model_cache))
            del model_cache[oldest_key]
            gc.collect()  # Force garbage collection
            logger.info(f"Evicted model {oldest_key} from cache")

class WeatherService:
    """Service to fetch weather data from Open-Meteo API"""
    
    @staticmethod
    def get_condition_from_code(weather_code: int) -> str:
        """Convert weather code to condition string"""
        if weather_code in [0]:
            return 'Clear'
        elif weather_code in [1, 2, 3]:
            return 'Partly Cloudy'
        elif weather_code in [45, 48]:
            return 'Foggy'
        elif weather_code in [51, 53, 55, 56, 57]:
            return 'Drizzle'
        elif weather_code in [61, 63, 65, 66, 67]:
            return 'Rain'
        elif weather_code in [71, 73, 75, 77]:
            return 'Snow'
        elif weather_code in [80, 81, 82]:
            return 'Rain Showers'
        elif weather_code in [95]:
            return 'Thunderstorm'
        else:
            return 'Unknown'
    
    @staticmethod
    def fetch_weather_for_timestamp(timestamp: datetime) -> Dict[str, float]:
        """Fetch weather data for a specific timestamp"""
        try:
            # Format timestamp for API (ISO format)
            start_date = timestamp.strftime('%Y-%m-%d')
            end_date = timestamp.strftime('%Y-%m-%d')
            
            url = (
                f"https://api.open-meteo.com/v1/forecast?"
                f"latitude={RUST_LATITUDE}&longitude={RUST_LONGITUDE}"
                f"&hourly=temperature_2m,precipitation,weather_code"
                f"&start_date={start_date}&end_date={end_date}"
                f"&timezone=Europe/Berlin"
            )
            
            logger.debug(f"Fetching weather for {timestamp}: {url}")
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Find the closest hour to our timestamp
            target_hour = timestamp.strftime('%Y-%m-%dT%H:00')
            
            if 'hourly' not in data or 'time' not in data['hourly']:
                logger.warning(f"No hourly data in weather response for {timestamp}")
                return {'temperature': 20.0, 'rain': 0.0}
            
            hourly_times = data['hourly']['time']
            
            # Find exact match or closest time
            if target_hour in hourly_times:
                index = hourly_times.index(target_hour)
            else:
                # Find closest hour
                target_dt = pd.to_datetime(target_hour)
                time_diffs = [abs((pd.to_datetime(t) - target_dt).total_seconds()) for t in hourly_times]
                index = time_diffs.index(min(time_diffs))
            
            temperature = data['hourly']['temperature_2m'][index] if index < len(data['hourly']['temperature_2m']) else 20.0
            rain = data['hourly']['precipitation'][index] if index < len(data['hourly']['precipitation']) else 0.0
            
            # Convert precipitation to binary rain indicator (>0.1mm = rain)
            rain_indicator = 1.0 if rain > 0.1 else 0.0
            
            logger.debug(f"Weather for {timestamp}: temp={temperature}°C, rain={rain}mm (indicator={rain_indicator})")
            
            return {
                'temperature': float(temperature),
                'rain': rain_indicator,
                'temperature_unscaled': float(temperature),
                'rain_unscaled': float(rain)
            }
            
        except Exception as e:
            logger.error(f"Error fetching weather for {timestamp}: {e}")
            # Return default values
            return {
                'temperature': 20.0,
                'rain': 0.0,
                'temperature_unscaled': 20.0,
                'rain_unscaled': 0.0
            }

class HolidayService:
    """Service to check holidays using the holidays package - matches training preprocessing exactly"""
    
    def __init__(self):
        # Initialize holiday date sets following the exact same logic as training
        self._initialize_holiday_dates()
    
    def _initialize_holiday_dates(self):
        """Initialize holiday date sets using the exact same logic as training preprocessing"""
        logger.info("Initializing holiday date sets using training preprocessing logic...")
        
        # Define year range (using current year and surrounding years for predictions)
        current_year = datetime.now().year
        years = range(current_year - 2, current_year + 3)  # Cover past and future years
        
        # Swiss cantons (exactly as in training code)
        swiss_cantons = ["ZH", "BE", "LU", "UR", "SZ", "OW", "NW", "GL", "ZG", "FR", "SO", "BS", "BL", "SH", "AR", "AI", "SG", "GR", "AG", "TG", "TI", "VD", "VS", "NE", "GE", "JU"]
        
        # Count Swiss holidays across all cantons
        swiss_holiday_count = {}
        for territory in swiss_cantons:
            swiss_holidays = holidays.country_holidays("CH", years=years, subdiv=territory)
            for _, name in swiss_holidays.items():
                if name not in swiss_holiday_count:
                    swiss_holiday_count[name] = 0
                swiss_holiday_count[name] += 1
        
        # Get common Swiss holidays (threshold: 70% of cantons have it)
        common_swiss_holidays = [name for name, count in swiss_holiday_count.items() if count >= 17]  # 70% of 26 cantons ≈ 17
        logger.info(f"Found {len(common_swiss_holidays)} common Swiss holidays")
        
        # Create Swiss holiday dates using Aargau (as per training code comment: "Aargau has all the holidays")
        self.swiss_holiday_dates = set()
        for holiday in common_swiss_holidays:
            for year in years:
                try:
                    ag_holiday_date = holidays.country_holidays("CH", years=year, subdiv="AG").get_named(holiday, lookup="exact")
                    if ag_holiday_date:
                        self.swiss_holiday_dates.add(ag_holiday_date[0])
                except Exception as e:
                    logger.debug(f"Could not find Swiss holiday {holiday} for {year}: {e}")
        
        # German states (exactly as in training code)
        german_states = ["BW", "BY", "BE", "BB", "HB", "HH", "HE", "MV", "NI", "NW", "RP", "SL", "SN", "ST", "SH", "TH"]
        
        # Count German holidays across all states
        german_holiday_count = {}
        for territory in german_states:
            german_holidays = holidays.country_holidays("DE", years=years, subdiv=territory)
            for _, name in german_holidays.items():
                if name not in german_holiday_count:
                    german_holiday_count[name] = 0
                german_holiday_count[name] += 1
        
        # Get common German holidays (threshold: 40 as per training code)
        common_german_holidays = [name for name, count in german_holiday_count.items() if count >= 40]
        logger.info(f"Found {len(common_german_holidays)} common German holidays")
        
        # Create German holiday dates using BW and HH fallback (as per training code)
        self.german_holiday_dates = set()
        for holiday in common_german_holidays:
            for year in years:
                try:
                    bw_holiday_date = holidays.country_holidays("DE", years=year, subdiv="BW").get_named(holiday, lookup="exact")
                    hh_holiday_date = holidays.country_holidays("DE", years=year, subdiv="HH").get_named(holiday, lookup="exact")
                    
                    if bw_holiday_date:
                        self.german_holiday_dates.add(bw_holiday_date[0])
                    elif hh_holiday_date:
                        self.german_holiday_dates.add(hh_holiday_date[0])
                except Exception as e:
                    logger.debug(f"Could not find German holiday {holiday} for {year}: {e}")
        
        # French territories (exactly as in training code)
        french_territories = ["BL", "GES", "GP", "GY", "MF", "MQ", "NC", "PF", "RE", "WF", "YT"]
        
        # Count French holidays across all territories
        french_holiday_count = {}
        for territory in french_territories:
            french_holidays = holidays.country_holidays("FR", years=years, subdiv=territory)
            for _, name in french_holidays.items():
                if name not in french_holiday_count:
                    french_holiday_count[name] = 0
                french_holiday_count[name] += 1
        
        # Get common French holidays (threshold: 30 as per training code)
        common_french_holidays = [name for name, count in french_holiday_count.items() if count >= 30]
        logger.info(f"Found {len(common_french_holidays)} common French holidays")
        
        # Create French holiday dates using Guadeloupe (as per training code)
        self.french_holiday_dates = set()
        for holiday in common_french_holidays:
            for year in years:
                try:
                    gp_holiday_date = holidays.country_holidays("FR", years=year, subdiv="GP").get_named(holiday, lookup="exact")
                    if gp_holiday_date:
                        self.french_holiday_dates.add(gp_holiday_date[0])
                except Exception as e:
                    logger.debug(f"Could not find French holiday {holiday} for {year}: {e}")
        
        logger.info(f"Holiday initialization complete:")
        logger.info(f"  - German holidays: {len(self.german_holiday_dates)} dates")
        logger.info(f"  - Swiss holidays: {len(self.swiss_holiday_dates)} dates")
        logger.info(f"  - French holidays: {len(self.french_holiday_dates)} dates")
    
    def check_holidays_for_date(self, date: datetime.date) -> Dict[str, int]:
        """Check if a date is a holiday in Germany, Switzerland, or France"""
        
        # Check if date is in the precomputed holiday sets (exactly like training)
        is_german_holiday = 1 if date in self.german_holiday_dates else 0
        is_swiss_holiday = 1 if date in self.swiss_holiday_dates else 0
        is_french_holiday = 1 if date in self.french_holiday_dates else 0
        
        logger.debug(f"Holidays for {date}: DE={is_german_holiday}, CH={is_swiss_holiday}, FR={is_french_holiday}")
        
        return {
            'is_german_holiday': is_german_holiday,
            'is_swiss_holiday': is_swiss_holiday,
            'is_french_holiday': is_french_holiday
        }

class FirestoreDataFetcher:
    """Fetches and preprocesses data from Firestore"""
    
    def __init__(self):
        self.weather_service = WeatherService()
        self.holiday_service = HolidayService()
    
    @staticmethod
    def get_firestore_attraction_name(attraction_id: str) -> Optional[str]:
        """Map frontend attraction ID to Firestore document ID"""
        mapping = next((m for m in LOCAL_RIDE_MAPPINGS if m["id"] == attraction_id), None)
        return mapping["name"] if mapping else NoneF
    
    def fetch_historical_data(self, attraction_name: str, hours_back: int = 48) -> pd.DataFrame:
        """Fetch historical wait time data from Firestore using CORRECT Python syntax"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=3)
    
        logger.info(f"🔍 DEBUGGING: Fetching historical data for '{attraction_name}' from {start_time} to {end_time}")
    
        attraction_doc_ref = db.collection("attractions").document(attraction_name)
        collection_ref = attraction_doc_ref.collection("queueTimes")
    
        # First, check if the collection exists
        try:
            recent_docs = list(collection_ref.order_by("__name__").limit(10).stream())
            if recent_docs:
                logger.info(f"🔍 DEBUGGING: Found {len(recent_docs)} documents in queueTimes collection")
                logger.info(f"🔍 DEBUGGING: Sample document IDs: {[doc.id for doc in recent_docs[:5]]}")
                sample_data = recent_docs[0].to_dict()
                logger.info(f"🔍 DEBUGGING: Sample document data: {sample_data}")
            else:
                logger.error(f"🔍 DEBUGGING: queueTimes collection is empty for {attraction_name}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"🔍 DEBUGGING: Error accessing queueTimes collection: {e}")
            return pd.DataFrame()
    
        all_data = []
        current_date = start_time.date()
        end_date = end_time.date()
        total_docs_found = 0
    
        while current_date <= end_date:
            try:
                year = current_date.year
                month = str(current_date.month).zfill(2)
                day = str(current_date.day).zfill(2)
                date_string = f"{year}{month}{day}"
            
                logger.info(f"🔍 DEBUGGING: Querying for date: {date_string}")
            
                # CORRECT Python Firestore syntax for document ID filtering
                start_doc_id = date_string
                end_doc_id = date_string + "z"
            
                # Use start_at and end_before instead of where() for document IDs
                query = collection_ref.order_by("__name__").start_at([start_doc_id]).end_before([end_doc_id])
                docs = list(query.stream())
            
                docs_processed = len(docs)
                total_docs_found += docs_processed
            
                logger.info(f"🔍 DEBUGGING: Found {docs_processed} documents for date {date_string}")
            
                day_data = []
                for doc in docs:
                    doc_id = doc.id
                    doc_data = doc.to_dict()
                
                    # Parse timestamp from document ID
                    try:
                        if len(doc_id) > 14:
                            timestamp = datetime.strptime(doc_id[:14], "%Y%m%d%H%M%S")
                        else:
                            timestamp = datetime.strptime(doc_id, "%Y%m%d%H%M%S")
                    except ValueError as e:
                        # Fallback to document timestamp field
                        if 'timestamp' in doc_data:
                            try:
                                if hasattr(doc_data['timestamp'], 'seconds'):
                                    timestamp = datetime.fromtimestamp(doc_data['timestamp'].seconds)
                                else:
                                    timestamp = pd.to_datetime(doc_data['timestamp'])
                            except Exception:
                                continue
                        else:
                            continue
                
                    # Filter by time range
                    if start_time <= timestamp <= end_time:
                        day_data.append({
                            'timestamp': timestamp,
                            'wait_time': doc_data.get('wait_time', 0),
                            'closed': 0
                        })
            
                all_data.extend(day_data)
                logger.info(f"🔍 DEBUGGING: Date {date_string}: Processed {docs_processed} docs, kept {len(day_data)} records")
                
            except Exception as e:
                logger.error(f"🔍 DEBUGGING: Error fetching data for date {current_date}: {e}")
                
            current_date += timedelta(days=1)
    
        logger.info(f"🔍 DEBUGGING: SUMMARY - Total docs found: {total_docs_found}, Total data points: {len(all_data)}")
    
        if not all_data:
            logger.error(f"🔍 DEBUGGING: No historical data found for '{attraction_name}'!")
            return pd.DataFrame()
    
        # Convert to DataFrame and process
        df = pd.DataFrame(all_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
    
        logger.info(f"🔍 DEBUGGING: Created DataFrame with {len(df)} records")
        logger.info(f"🔍 DEBUGGING: Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
        processed_df = self._process_to_30_minute_intervals(df, attraction_name)
        logger.info(f"🔍 DEBUGGING: After processing: {len(processed_df)} records")
    
        return processed_df
    
    def _process_to_30_minute_intervals(self, df: pd.DataFrame, attraction_name: str) -> pd.DataFrame:
        """Process 5-minute data into 30-minute intervals and add weather/holiday data"""
        if df.empty:
            return df
        
        logger.info("Processing data into 30-minute intervals...")
        
        # Filter for park operating hours (roughly 9 AM to 9 PM)
        df_filtered = df[
            (df['timestamp'].dt.hour >= 9) & 
            (df['timestamp'].dt.hour <= 21)
        ].copy()
        
        if df_filtered.empty:
            logger.warning("No data found during operating hours")
            return pd.DataFrame()
        
        # Create 30-minute time bins
        df_filtered['time_30min'] = df_filtered['timestamp'].dt.floor('30T')
        
        # Group by 30-minute intervals and calculate averages
        interval_data = []
        
        for time_bin, group in df_filtered.groupby('time_30min'):
            # Calculate average wait time for this 30-minute interval
            # Only include non-closed entries for wait time calculation
            open_entries = group[group['closed'] == 0]
            
            if len(open_entries) > 0:
                avg_wait_time = open_entries['wait_time'].mean()
                # Check if any entry in this interval was closed
                any_closed = group['closed'].max()
            else:
                # All entries were closed
                avg_wait_time = 0
                any_closed = 1
            
            # Get weather data for this timestamp
            weather_data = self.weather_service.fetch_weather_for_timestamp(time_bin)
            
            # Get holiday information for this date
            holiday_data = self.holiday_service.check_holidays_for_date(time_bin.date())
            
            interval_data.append({
                'timestamp': time_bin,
                'wait_time': round(avg_wait_time, 1),
                'closed': any_closed,
                'data_points_count': len(group),  # How many 5-min intervals we averaged
                **weather_data,
                **holiday_data
            })
        
        if not interval_data:
            logger.warning("No 30-minute intervals created")
            return pd.DataFrame()
        
        # Convert to DataFrame
        processed_df = pd.DataFrame(interval_data)
        processed_df['timestamp'] = pd.to_datetime(processed_df['timestamp'])
        processed_df = processed_df.sort_values('timestamp').reset_index(drop=True)
        
        # Keep only the most recent 48 intervals (24 hours of 30-min intervals)
        if len(processed_df) > 48:
            processed_df = processed_df.tail(48).reset_index(drop=True)
        
        logger.info(f"Created {len(processed_df)} 30-minute intervals for model input")
        logger.info(f"Time range: {processed_df['timestamp'].min()} to {processed_df['timestamp'].max()}")
        
        return processed_df
    
    def preprocess_for_prediction(self, df: pd.DataFrame, ride_name: str) -> pd.DataFrame:
        """Preprocess data to match training format"""
        if df.empty:
            return df
        
        # Add temporal features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = (df['timestamp'].dt.dayofweek >= 5).astype(int)
        
        # Add cyclical encodings
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['weekday_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['weekday_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['minute_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.minute / 60)
        df['minute_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.minute / 60)
        
        # Add part of day features
        df['part_of_day_morning'] = ((df['hour'] >= 6) & (df['hour'] < 12)).astype(int)
        df['part_of_day_afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 18)).astype(int)
        df['part_of_day_evening'] = ((df['hour'] >= 18) & (df['hour'] < 22)).astype(int)
        df['part_of_day_night'] = ((df['hour'] >= 22) | (df['hour'] < 6)).astype(int)
        
        # Add season features
        month = df['month'].iloc[0] if not df.empty else 6
        df['season_spring'] = int(month in [3, 4, 5])
        df['season_summer'] = int(month in [6, 7, 8])
        df['season_fall'] = int(month in [9, 10, 11])
        df['season_winter'] = int(month in [12, 1, 2])
        
        # Add year features
        year = df['timestamp'].dt.year.iloc[0] if not df.empty else 2024
        for y in range(2017, 2025):
            df[f'year_{y}'] = int(year == y)
        
        # Add weekday feature
        df['weekday'] = df['day_of_week']
        
        logger.info(f"Preprocessed data with {len(df.columns)} features for {len(df)} records")
        return df

    def generate_future_data(self, start_timestamp: pd.Timestamp, prediction_steps: int) -> pd.DataFrame:
        """Generate future data with weather and holiday information (30-minute intervals)"""
        # Generate future timestamps with 30-minute intervals
        future_timestamps = pd.date_range(
            start=start_timestamp,
            periods=prediction_steps,
            freq='30T'  # Changed from 15T to 30T
        )
        
        future_data = []
        
        for ts in future_timestamps:
            # Get weather data for this timestamp
            weather_data = self.weather_service.fetch_weather_for_timestamp(ts)
            
            # Get holiday information
            holiday_data = self.holiday_service.check_holidays_for_date(ts.date())
            
            future_data.append({
                'timestamp': ts,
                'wait_time': 0,  # Will be predicted
                'closed': 0,     # Assume open for predictions
                **weather_data,
                **holiday_data
            })
        
        df = pd.DataFrame(future_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Apply same preprocessing
        processed_df = self.preprocess_for_prediction(df, "")  # ride_name not needed for feature generation
        
        logger.info(f"Generated {len(df)} future data points (30-min intervals) starting from {start_timestamp}")
        return processed_df

# Initialize services
model_manager = ModelManager()
data_fetcher = FirestoreDataFetcher()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Basic service health
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "models_cached": len(model_cache),
            "service": "queue-prediction-service",
            "region": "europe-west6"
        }
        
        # Test basic imports
        import torch
        health_status["torch_available"] = True
        health_status["cuda_available"] = torch.cuda.is_available()
        
        # Test storage access
        try:
            bucket = storage_client.bucket(MODEL_BUCKET)
            bucket.get_blob("models/")
            health_status["storage_accessible"] = True
        except Exception as e:
            health_status["storage_accessible"] = False
            health_status["storage_error"] = str(e)
        
        # Test weather API
        try:
            test_weather = data_fetcher.weather_service.fetch_weather_for_timestamp(datetime.now())
            health_status["weather_api_accessible"] = True
        except Exception as e:
            health_status["weather_api_accessible"] = False
            health_status["weather_error"] = str(e)
        
        # Test holidays
        try:
            test_holidays = data_fetcher.holiday_service.check_holidays_for_date(datetime.now().date())
            health_status["holidays_accessible"] = True
        except Exception as e:
            health_status["holidays_accessible"] = False
            health_status["holidays_error"] = str(e)
        
        return jsonify(health_status)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/predict', methods=['POST'])
def predict_wait_times():
    """Main prediction endpoint"""
    try:
        # Parse request
        data = request.get_json()
        attraction_id = data.get('attraction_id')
        prediction_hours = data.get('prediction_hours', 24)  # Default 24 hours
        start_time = data.get('start_time')  # Optional custom start time
        
        if not attraction_id:
            return jsonify({"error": "attraction_id is required"}), 400
        
        # Map to Firestore attraction name
        attraction_name = data_fetcher.get_firestore_attraction_name(attraction_id)
        if not attraction_name:
            return jsonify({"error": f"Unknown attraction_id: {attraction_id}"}), 400
        
        logger.info(f"Processing prediction request for {attraction_name}")
        
        # Fetch historical data (now with dynamic weather and holidays)
        historical_df = data_fetcher.fetch_historical_data(attraction_name, hours_back=48)
        
        if historical_df.empty:
            return jsonify({"error": "No historical data available"}), 404
        
        # Preprocess historical data
        processed_df = data_fetcher.preprocess_for_prediction(historical_df, attraction_name)
        
        # Get model
        model = model_manager.get_model(attraction_name)
        
        # Prepare data for prediction
        prediction_df = model.preprocess_input(processed_df)
        
        # Generate predictions
        prediction_steps = int(prediction_hours * 2)  # 30-minute intervals (changed from * 4)
        
        # Determine start time (round to next 30-minute interval)
        if start_time:
            start_timestamp = pd.to_datetime(start_time)
        else:
            start_timestamp = pd.Timestamp.now().ceil('30T')  # Changed from 15T to 30T
        
        # Generate future data with dynamic weather and holidays
        future_df = data_fetcher.generate_future_data(start_timestamp, prediction_steps)
        
        # Prepare future static features
        future_features = model.preprocess_input(future_df)
        future_static_features = future_features[model.static_feature_cols].values
        
        # Get recent sequence for autoregressive prediction
        recent_data = prediction_df.tail(model.seq_length).copy()
        if len(recent_data) < model.seq_length:
            # Pad with averages if not enough data
            avg_wait_time = processed_df['wait_time'].mean() if not processed_df.empty else 10.0
            padding_needed = model.seq_length - len(recent_data)
            
            # Create padding data
            last_timestamp = recent_data['timestamp'].iloc[-1] if not recent_data.empty else pd.Timestamp.now()
            for i in range(padding_needed):
                padding_data = {
                    'timestamp': [last_timestamp - pd.Timedelta(minutes=30*(i+1))],  # Changed from 15 to 30
                    'wait_time': [avg_wait_time]
                }
                # Add static features with defaults
                for col in model.static_feature_cols:
                    if col in recent_data.columns:
                        padding_data[col] = [recent_data[col].iloc[-1]]
                    else:
                        padding_data[col] = [0]
                
                recent_data = pd.concat([recent_data, pd.DataFrame(padding_data)], ignore_index=True)
        
        # Build initial sequence
        initial_history = []
        gb_model = model.gb_model
        
        for _, row in recent_data.tail(model.seq_length).iterrows():
            wait_time = row['wait_time']
            static_features = np.array([row[col] for col in model.static_feature_cols])
            gb_pred = gb_model.predict(static_features.reshape(1, -1))[0]
            residual = wait_time - gb_pred
            initial_history.append((wait_time, residual))
        
        # Generate predictions
        initial_static = future_static_features[0] if len(future_static_features) > 0 else np.zeros(len(model.static_feature_cols))
        
        predictions = model.predict_sequence(
            initial_static_features=initial_static,
            initial_history=initial_history,
            future_static_features=future_static_features,
            horizon=min(prediction_steps, len(future_static_features))
        )
        
        # Format response with numpy type conversion
        response = {
            "attraction_id": attraction_id,
            "attraction_name": attraction_name,
            "start_time": start_timestamp.isoformat(),
            "prediction_hours": prediction_hours,
            "predictions": []
        }
        
        for i, pred in enumerate(predictions):
            prediction_data = {
                "timestamp": future_df.iloc[i]['timestamp'].isoformat() if i < len(future_df) else start_timestamp.isoformat(),
                "predicted_wait_time": float(max(0, pred["wait_time_prediction"])),  # Convert to Python float
                "baseline_prediction": float(pred["baseline_prediction"]),           # Convert to Python float
                "residual_prediction": float(pred["residual_prediction"]),           # Convert to Python float
                "temperature": float(future_df.iloc[i]['temperature']) if i < len(future_df) else None,
                "rain": float(future_df.iloc[i]['rain']) if i < len(future_df) else None,
                "is_holiday": bool(any([  # Convert to Python bool
                    future_df.iloc[i]['is_german_holiday'] if i < len(future_df) else 0,
                    future_df.iloc[i]['is_swiss_holiday'] if i < len(future_df) else 0,
                    future_df.iloc[i]['is_french_holiday'] if i < len(future_df) else 0
                ]))
            }
            response["predictions"].append(prediction_data)
        
        # Apply additional conversion to ensure all numpy types are converted
        response = convert_numpy_types(response)
        
        logger.info(f"Successfully generated {len(predictions)} predictions for {attraction_name}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/models/status', methods=['GET'])
def model_status():
    """Get status of loaded models"""
    return jsonify({
        "cached_models": list(model_cache.keys()),
        "cache_size": len(model_cache),
        "max_cache_size": MAX_MODELS_IN_MEMORY
    })

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint for weekly predictions"""
    try:
        data = request.get_json() or {}
        prediction_days = data.get('prediction_days', 7)
        trigger_source = data.get('trigger_source', 'manual')
        
        logger.info(f"Starting batch predictions for {prediction_days} days (triggered by: {trigger_source})")
        
        # Import here to avoid circular imports
        from batch_predictor import BatchPredictor
        
        predictor = BatchPredictor()
        results = predictor.run_weekly_predictions(prediction_days)
        
        # Count successful predictions
        successful = sum(1 for r in results.values() if r["status"] == "success")
        total = len(results)
        
        response = {
            "status": "completed",
            "trigger_source": trigger_source,
            "prediction_days": prediction_days,
            "total_attractions": total,
            "successful_attractions": successful,
            "failed_attractions": total - successful,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Batch predictions completed: {successful}/{total} attractions successful")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/predictions/<attraction_id>', methods=['GET'])
def get_predictions(attraction_id: str):
    """Retrieve saved predictions for a specific attraction and date"""
    try:
        date_str = request.args.get('date')  # Format: YYYY-MM-DD
        
        if not date_str:
            return jsonify({"error": "date parameter required (format: YYYY-MM-DD)"}), 400
        
        # Parse date
        try:
            target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400
        
        # Map attraction ID to Firestore name
        attraction_name = data_fetcher.get_firestore_attraction_name(attraction_id)
        if not attraction_name:
            return jsonify({"error": f"Unknown attraction_id: {attraction_id}"}), 400
        
        # Import here to avoid circular imports
        from batch_predictor import BatchPredictor
        
        predictor = BatchPredictor()
        predictions = predictor.get_predictions_for_attraction(attraction_name, target_date)
        
        if not predictions:
            return jsonify({
                "attraction_id": attraction_id,
                "attraction_name": attraction_name,
                "date": date_str,
                "predictions": [],
                "message": "No predictions found for this date. Run batch predictions first."
            })
        
        response = {
            "attraction_id": attraction_id,
            "attraction_name": attraction_name,
            "date": date_str,
            "predictions_count": len(predictions),
            "predictions": predictions
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error retrieving predictions: {str(e)}", exc_info=True)
        return jsonify({"error": f"Failed to retrieve predictions: {str(e)}"}), 500

@app.route('/models/clear-cache', methods=['POST'])
def clear_model_cache():
    """Clear model cache (for debugging)"""
    with cache_lock:
        model_cache.clear()
        gc.collect()
    
    return jsonify({"message": "Model cache cleared"})

@app.route('/debug/firestore', methods=['GET'])
def debug_firestore():
    """Enhanced debug endpoint with better error handling"""
    try:
        attraction_id = request.args.get('attraction_id', 'silver-star')
        attraction_name = data_fetcher.get_firestore_attraction_name(attraction_id)
        
        if not attraction_name:
            return jsonify({"error": f"Unknown attraction_id: {attraction_id}"}), 400
        
        debug_info = {
            "attraction_id": attraction_id,
            "attraction_name": attraction_name,
            "debug_timestamp": datetime.now().isoformat(),
            "mapping_used": next((m for m in LOCAL_RIDE_MAPPINGS if m["id"] == attraction_id), None)
        }
        
        # Test basic Firestore connection
        try:
            # Try to access the database
            debug_info["firestore_accessible"] = True
            logger.info("🔍 DEBUG: Basic Firestore connection successful")
        except Exception as e:
            debug_info["firestore_accessible"] = False
            debug_info["firestore_error"] = str(e)
            logger.error(f"🔍 DEBUG: Firestore connection failed: {e}")
            return jsonify(debug_info), 500
        
        # Try to list attractions collection
        try:
            attractions_ref = db.collection("attractions")
            logger.info("🔍 DEBUG: Attempting to list attractions...")
            
            # Try different approaches to list attractions
            attractions_list = []
            
            # Approach 1: Simple limit query
            try:
                docs = list(attractions_ref.limit(20).stream())
                logger.info(f"🔍 DEBUG: Found {len(docs)} attractions using limit query")
                attractions_list = docs
            except Exception as e:
                logger.error(f"🔍 DEBUG: Limit query failed: {e}")
                debug_info["limit_query_error"] = str(e)
            
            # If limit query failed, try get specific document
            if not attractions_list:
                try:
                    specific_doc = attractions_ref.document(attraction_name).get()
                    if specific_doc.exists:
                        attractions_list = [specific_doc]
                        logger.info(f"🔍 DEBUG: Found specific attraction: {attraction_name}")
                    else:
                        logger.info(f"🔍 DEBUG: Specific attraction {attraction_name} does not exist")
                except Exception as e:
                    logger.error(f"🔍 DEBUG: Specific document query failed: {e}")
                    debug_info["specific_doc_error"] = str(e)
            
            # Process found attractions
            available_attractions = []
            for doc in attractions_list:
                try:
                    doc_data = doc.to_dict()
                    attraction_info = {
                        "id": doc.id,
                        "data_keys": list(doc_data.keys()) if doc_data else [],
                        "has_data": bool(doc_data)
                    }
                    
                    # Test queue times access
                    try:
                        queue_ref = doc.reference.collection("queueTimes")
                        recent_docs = list(queue_ref.order_by("__name__").limit(3).stream())
                        
                        attraction_info.update({
                            "has_queue_times": len(recent_docs) > 0,
                            "queue_docs_count": len(recent_docs),
                            "sample_queue_doc_ids": [d.id for d in recent_docs]
                        })
                        
                        if recent_docs:
                            sample_data = recent_docs[0].to_dict()
                            attraction_info["sample_queue_data"] = sample_data
                            
                    except Exception as e:
                        attraction_info["queue_times_error"] = str(e)
                    
                    available_attractions.append(attraction_info)
                    
                except Exception as e:
                    logger.error(f"🔍 DEBUG: Error processing doc {doc.id}: {e}")
                    available_attractions.append({
                        "id": doc.id,
                        "error": str(e)
                    })
            
            debug_info["available_attractions"] = available_attractions
            debug_info["total_attractions"] = len(available_attractions)
            
            # Check permissions by trying different operations
            try:
                # Test if we can read from the attractions collection
                test_query = attractions_ref.limit(1)
                test_docs = list(test_query.stream())
                debug_info["can_read_attractions"] = True
                debug_info["test_query_result"] = len(test_docs)
            except Exception as e:
                debug_info["can_read_attractions"] = False
                debug_info["read_error"] = str(e)
                logger.error(f"🔍 DEBUG: Cannot read attractions collection: {e}")
            
            # Specific attraction test
            try:
                attraction_ref = attractions_ref.document(attraction_name)
                attraction_doc = attraction_ref.get()
                
                debug_info["specific_attraction"] = {
                    "exists": attraction_doc.exists,
                    "id": attraction_name,
                    "can_access": True
                }
                
                if attraction_doc.exists:
                    debug_info["specific_attraction"]["data"] = attraction_doc.to_dict()
                    
                    # Test queue times for specific attraction
                    try:
                        queue_ref = attraction_ref.collection("queueTimes")
                        
                        # Test the exact same query used in the main code
                        recent_docs = list(queue_ref.order_by("__name__").limit(5).stream())
                        
                        debug_info["specific_queue_test"] = {
                            "collection_accessible": True,
                            "docs_found": len(recent_docs),
                            "sample_doc_ids": [doc.id for doc in recent_docs]
                        }
                        
                        if recent_docs:
                            # Test date-based query
                            today_string = datetime.now().strftime("%Y%m%d")
                            try:
                                date_query = queue_ref.order_by("__name__").start_at([today_string]).end_before([today_string + "z"])
                                date_docs = list(date_query.stream())
                                
                                debug_info["date_query_test"] = {
                                    "date_string": today_string,
                                    "docs_found": len(date_docs),
                                    "query_successful": True
                                }
                            except Exception as e:
                                debug_info["date_query_test"] = {
                                    "date_string": today_string,
                                    "query_successful": False,
                                    "error": str(e)
                                }
                        
                    except Exception as e:
                        debug_info["specific_queue_test"] = {
                            "collection_accessible": False,
                            "error": str(e)
                        }
                        
            except Exception as e:
                debug_info["specific_attraction"] = {
                    "can_access": False,
                    "error": str(e)
                }
                
        except Exception as e:
            debug_info["attractions_collection_error"] = str(e)
            debug_info["attractions_error_type"] = type(e).__name__
            logger.error(f"🔍 DEBUG: Cannot access attractions collection: {e}")
        
        # Add service account info
        try:
            # Try to determine which service account we're using
            import requests
            metadata_url = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email"
            headers = {"Metadata-Flavor": "Google"}
            response = requests.get(metadata_url, headers=headers, timeout=5)
            if response.status_code == 200:
                debug_info["service_account"] = response.text
            else:
                debug_info["service_account"] = "unknown"
        except Exception as e:
            debug_info["service_account"] = f"error: {str(e)}"
        
        return jsonify(debug_info)
        
    except Exception as e:
        logger.error(f"Debug endpoint error: {e}")
        return jsonify({
            "error": str(e),
            "error_type": type(e).__name__,
            "debug_timestamp": datetime.now().isoformat()
        }), 500
        
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)