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
from single_day_predictor import SingleDayPredictor

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
    {"id": "blue-fire", "name": "blue fire Megacoaster", "normalized_name": "Blue_Fire_Megacoaster"},
    {"id": "wodan", "name": "WODAN - Timburcoaster", "normalized_name": "Wodan_Timburcoaster"},
    {"id": "voletarium", "name": "Voletarium", "normalized_name": "Voletarium"},
    {"id": "alpine-express-enzian", "name": "Alpine Express 'Enzian'", "normalized_name": "Alpine_Express_Enzian"},
    {"id": "arena-of-football", "name": "Arena of Football - Be Part of It!", "normalized_name": "Arena_of_Football_Be_Part_of_It"},
    {"id": "arthur", "name": "ARTHUR", "normalized_name": "Arthur"},
    {"id": "atlantica-supersplash", "name": "Atlantica SuperSplash", "normalized_name": "Atlantica_SuperSplash"},
    {"id": "atlantis-adventure", "name": "Atlantis Adventure", "normalized_name": "Atlantis_Adventure"},
    {"id": "baaa-express", "name": "Ba-a-a Express", "normalized_name": "Baaa_Express"},
    {"id": "bellevue-ferris-wheel", "name": "Bellevue Ferris Wheel", "normalized_name": "Bellevue_Ferris_Wheel"},
    {"id": "castello-dei-medici", "name": "Castello dei Medici", "normalized_name": "Castello_dei_Medici"},
    {"id": "dancing-dingie", "name": "Dancing Dingie", "normalized_name": "Dancing_Dingie"},
    {"id": "euromir", "name": "Euro-Mir", "normalized_name": "Euro_Mir"},
    {"id": "eurosat-cancan-coaster", "name": "Eurosat - CanCan Coaster", "normalized_name": "Eurosat_CanCan_Coaster"},
    {"id": "eurotower", "name": "Euro-Tower", "normalized_name": "Euro_Tower"},
    {"id": "fjordrafting", "name": "Fjord-Rafting", "normalized_name": "Fjord_Rafting"},
    {"id": "jim-button-journey", "name": "Jim Button - Journey through Morrowland", "normalized_name": "Jim_Button_Journey_Through_Morrowland"},
    {"id": "josefinas-imperial-journey", "name": "Josefina's Magical Imperial Journey", "normalized_name": "Josefinas_Magical_Imperial_Journey"},
    {"id": "kolumbusjolle", "name": "Kolumbusjolle", "normalized_name": "Kolumbusjolle"},
    {"id": "madame-freudenreich", "name": "Madame Freudenreich Curiosités", "normalized_name": "Madame_Freudenreich_Curiosites"},
    {"id": "matterhorn-blitz", "name": "Matterhorn-Blitz", "normalized_name": "Matterhorn_Blitz"},
    {"id": "old-macdonald", "name": "Old Mac Donald's Tractor Fun", "normalized_name": "Old_Mac_Donalds_Tractor_Fun"},
    {"id": "pegasus", "name": "Pegasus", "normalized_name": "Pegasus"},
    {"id": "pirates-in-batavia", "name": "Pirates in Batavia", "normalized_name": "Pirates_in_Batavia"},
    {"id": "poppy-towers", "name": "Poppy Towers", "normalized_name": "Poppy_Towers"},
    {"id": "poseidon", "name": "Water rollercoaster Poseidon", "normalized_name": "Poseidon"},
    {"id": "snorri-touren", "name": "Snorri Touren", "normalized_name": "Snorri_Touren"},
    {"id": "swiss-bob-run", "name": "Swiss Bob Run", "normalized_name": "Swiss_Bob_Run"},
    {"id": "tirol-log-flume", "name": "Tirol Log Flume", "normalized_name": "Tirol_Log_Flume"},
    {"id": "tnnevirvel", "name": "Tnnevirvel", "normalized_name": "Tnnevirvel"},
    {"id": "vienna-wave-swing", "name": "Vienna Wave Swing - 'Glückspilz'", "normalized_name": "Vienna_Wave_Swing_Glckspilz"},
    {"id": "vindjammer", "name": "Vindjammer", "normalized_name": "Vindjammer"},
    {"id": "volo-da-vinci", "name": "Volo da Vinci", "normalized_name": "Volo_da_Vinci"},
    {"id": "voltron-nevera", "name": "Voltron Nevera powered by Rimac", "normalized_name": "Voltron_Nevera_Powered_by_Rimac"},
    {"id": "whale-adventures", "name": "Whale Adventures - Northern Lights", "normalized_name": "Whale_Adventures_Northern_Lights"}
]

MODEL_FILE_MAPPINGS = {
    # Map from Firestore attraction names to model file names
    "ARTHUR": "arthur",
    "Alpine Express 'Enzian'": "alpine_express_enzian",
    "Arena of Football - Be Part of It!": "arena_of_football__be_part_of_it",  # Note: double underscore
    "Atlantica SuperSplash": "atlantica_supersplash", 
    "Atlantis Adventure": "atlantis_adventure",
    "Ba-a-a Express": "baaa_express",
    "Bellevue Ferris Wheel": "bellevue_ferris_wheel",
    "Castello dei Medici": "castello_dei_medici",
    "Dancing Dingie": "dancing_dingie",
    "Euro-Mir": "euromir",
    "Euro-Tower": "eurotower",
    "Eurosat - CanCan Coaster": "eurosat__cancan_coaster",  # Note: double underscore
    "Fjord-Rafting": "fjordrafting", 
    "Jim Button - Journey through Morrowland": "jim_button__journey_through_morrowland",
    "Josefina's Magical Imperial Journey": "josefinas_magical_imperial_journey",
    "Kolumbusjolle": "kolumbusjolle",
    "Madame Freudenreich Curiosités": "madame_freudenreich_curiosits",  # Note: no 'e' in curiosits
    "Matterhorn-Blitz": "matterhornblitz",
    "Old Mac Donald's Tractor Fun": "old_mac_donalds_tractor_fun",
    "Pegasus": "pegasus",
    "Pirates in Batavia": "pirates_in_batavia",
    "Poppy Towers": "poppy_towers",
    "Water rollercoaster Poseidon": "poseidon",
    "Silver Star": "silver_star",
    "Snorri Touren": "snorri_touren",
    "Swiss Bob Run": "swiss_bob_run",
    "Tirol Log Flume": "tirol_log_flume",
    "Tnnevirvel": "tnnevirvel",
    "Vienna Wave Swing - 'Glückspilz'": "vienna_wave_swing_glckspilz",  # Note: no umlaut
    "Vindjammer": "vindjammer",
    "Voletarium": "voletarium",
    "Volo da Vinci": "volo_da_vinci",
    "Voltron Nevera powered by Rimac": "voltron_nevera_powered_by_rimac",
    "WODAN - Timburcoaster": "wodan_timburcoaster",
    "Whale Adventures - Northern Lights": "whale_adventures_northern_lights",
    "blue fire Megacoaster": "blue_fire_megacoaster"
}


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
        """Load model files from Cloud Storage using the correct model file naming - ENHANCED DEBUGGING"""
    
        logger.info(f"🔍 ATTEMPTING MODEL LOAD for '{ride_name}'")
    
        # Use the MODEL_FILE_MAPPINGS for model files (not LOCAL_RIDE_MAPPINGS)
        if ride_name in MODEL_FILE_MAPPINGS:
            normalized_name = MODEL_FILE_MAPPINGS[ride_name]
            logger.info(f"✅ Found model mapping for '{ride_name}' -> '{normalized_name}'")
        else:
            logger.error(f"❌ No model file mapping found for '{ride_name}'")
            logger.error(f"Available mappings ({len(MODEL_FILE_MAPPINGS)}): {list(MODEL_FILE_MAPPINGS.keys())}")
        
            # Try to find similar names
            similar_names = [name for name in MODEL_FILE_MAPPINGS.keys() if ride_name.lower() in name.lower() or name.lower() in ride_name.lower()]
            if similar_names:
                logger.error(f"Similar names found: {similar_names}")
        
            raise ValueError(f"No model file mapping found for attraction: {ride_name}")
    
        logger.info(f"Loading model for '{ride_name}' using model file name: '{normalized_name}'")
    
        # Download model files to temp directory
        temp_dir = f"/tmp/{normalized_name}"
        os.makedirs(temp_dir, exist_ok=True)
    
        # Download GB model
        gb_blob_path = f"models/{normalized_name}_gb_baseline.pkl"
        gb_blob = self.bucket.blob(gb_blob_path)
        gb_path = f"{temp_dir}/{normalized_name}_gb_baseline.pkl"
    
        logger.info(f"🔄 Downloading GB model from: {gb_blob_path}")
    
        try:
            # Check if blob exists first
            if not gb_blob.exists():
                logger.error(f"❌ GB model blob does not exist: {gb_blob_path}")
                # List similar files for debugging
                try:
                    blobs = list(self.bucket.list_blobs(prefix=f"models/{normalized_name}"))
                    similar_files = [blob.name for blob in blobs]
                    logger.error(f"Files starting with 'models/{normalized_name}': {similar_files}")
                
                    # Also list all GB models
                    all_gb_blobs = list(self.bucket.list_blobs(prefix="models/"))
                    all_gb_models = [blob.name for blob in all_gb_blobs if blob.name.endswith('_gb_baseline.pkl')]
                    logger.error(f"All available GB models: {all_gb_models}")
                except Exception as list_e:
                    logger.error(f"Could not list available models: {list_e}")
                raise FileNotFoundError(f"GB model blob not found: {gb_blob_path}")
        
            gb_blob.download_to_filename(gb_path)
            logger.info(f"✅ Successfully downloaded GB model to {gb_path}")
        except Exception as e:
            logger.error(f"❌ Failed to download GB model: {e}")
            raise
    
        # Download TCN model
        tcn_blob_path = f"models/{normalized_name}_cached_scheduled_sampling_tcn.pt"
        tcn_blob = self.bucket.blob(tcn_blob_path)
        tcn_path = f"{temp_dir}/{normalized_name}_cached_scheduled_sampling_tcn.pt"
    
        logger.info(f"🔄 Downloading TCN model from: {tcn_blob_path}")
    
        try:
            # Check if blob exists first
            if not tcn_blob.exists():
                logger.error(f"❌ TCN model blob does not exist: {tcn_blob_path}")
                # List similar files for debugging
                try:
                    blobs = list(self.bucket.list_blobs(prefix=f"models/{normalized_name}"))
                    similar_files = [blob.name for blob in blobs]
                    logger.error(f"Files starting with 'models/{normalized_name}': {similar_files}")
                
                    # Also list all TCN models
                    all_tcn_blobs = list(self.bucket.list_blobs(prefix="models/"))
                    all_tcn_models = [blob.name for blob in all_tcn_blobs if blob.name.endswith('_cached_scheduled_sampling_tcn.pt')]
                    logger.error(f"All available TCN models: {all_tcn_models}")
                except Exception as list_e:
                    logger.error(f"Could not list available models: {list_e}")
                raise FileNotFoundError(f"TCN model blob not found: {tcn_blob_path}")
        
            tcn_blob.download_to_filename(tcn_path)
            logger.info(f"✅ Successfully downloaded TCN model to {tcn_path}")
        except Exception as e:
            logger.error(f"❌ Failed to download TCN model: {e}")
            raise
    
        # Initialize predictor
        try:
            predictor = WaitTimePredictor(ride_name, temp_dir)
        
            # CRITICAL: Test feature compatibility immediately
            logger.info(f"🔍 Testing feature compatibility for '{ride_name}'")
        
            # Create a small test dataset to verify features
            test_timestamp = pd.Timestamp("2025-06-10 15:30:00")
            test_data = pd.DataFrame({
                'timestamp': [test_timestamp],
                'wait_time': [15.0],
                'closed': [0],
                'temperature': [20.0],
                'rain': [0.0],
                'temperature_unscaled': [20.0],
                'rain_unscaled': [0.0],
                'is_german_holiday': [0],
                'is_swiss_holiday': [0], 
            '   is_french_holiday': [0]
            })
        
            try:
                # Test preprocessing
                processed_test = predictor.preprocess_input(test_data)
                logger.info(f"✅ Feature preprocessing successful for '{ride_name}'")
                logger.info(f"Features created: {len(predictor.static_feature_cols)}")
                logger.info(f"Expected features: {predictor.config['static_features_size']}")
            
                # Log first few features for debugging
                if len(predictor.static_feature_cols) > 0:
                    logger.info(f"First 10 features: {predictor.static_feature_cols[:10]}")
                
            except Exception as feature_error:
                logger.error(f"❌ Feature preprocessing failed for '{ride_name}': {feature_error}")
                raise
        
            logger.info(f"✅ Successfully loaded and tested predictor for '{ride_name}'")
            return predictor
        
        except Exception as e:
            logger.error(f"❌ Failed to create predictor for '{ride_name}': {e}")
            raise
    
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
        """
        Preprocess data to match training format EXACTLY
        This replaces the simple preprocessing with the full training pipeline
        """
        if df.empty:
            return df
        
        logger.info(f"Starting comprehensive preprocessing for {ride_name}")
        
        # Import the corrected preprocessing function
        from datasets.data_utils import preprocess_data
        
        # Apply the EXACT same preprocessing as training
        processed_df = preprocess_data(df, ride_name)
        
        logger.info(f"Comprehensive preprocessing completed: {len(processed_df.columns)} features for {len(processed_df)} records")
        
        # Log some key features to verify
        key_features = ['closed', 'is_german_holiday', 'is_swiss_holiday', 'is_french_holiday', 
                       'weekday', 'is_weekend', 'temperature', 'rain', 'temperature_unscaled', 'rain_unscaled']
        
        for feature in key_features:
            if feature in processed_df.columns:
                logger.debug(f"Feature {feature}: min={processed_df[feature].min():.3f}, max={processed_df[feature].max():.3f}")
        
        return processed_df

    def generate_future_data(self, start_timestamp: pd.Timestamp, prediction_steps: int, attraction_name: str = "") -> pd.DataFrame:
        """Generate future data with weather and holiday information (30-minute intervals, 09:00-17:30 CEST)"""
        import pytz
    
        # Define CEST timezone
        cest_tz = pytz.timezone('Europe/Zurich')
    
        future_data = []
        current_time = start_timestamp
        steps_generated = 0
    
        logger.info(f"Generating {prediction_steps} future data points (30-min intervals, 09:00-17:30 CEST)")

        while steps_generated < prediction_steps:
            # Convert to CEST for hour checking
            current_time_cest = current_time.tz_localize('UTC').astimezone(cest_tz) if current_time.tz is None else current_time.astimezone(cest_tz)
            hour = current_time_cest.hour
            minute = current_time_cest.minute

            # ENFORCED: Operating hours 09:00 to 17:30 in CEST
            is_operating_hours = (hour >= 9 and hour < 17) or (hour == 17 and minute <= 30)
        
            logger.debug(f"Time {current_time_cest.strftime('%H:%M')} CEST: operating_hours={is_operating_hours}")

            if is_operating_hours:
                # Get weather data for this timestamp
                weather_data = self.weather_service.fetch_weather_for_timestamp(current_time)

                # Get holiday information
                holiday_data = self.holiday_service.check_holidays_for_date(current_time.date())

                future_data.append({
                    'timestamp': current_time,
                    'wait_time': 0,  # Will be predicted
                    'closed': 0,     # Assume open for predictions
                    **weather_data,
                    **holiday_data
                })

            steps_generated += 1

            # Move to next 30-minute interval
            current_time += pd.Timedelta(minutes=30)

            # Skip to next day if past operating hours (17:30 CEST)
            if hour >= 17 and minute > 30:
                # Move to next day at 09:00 CEST
                next_day = current_time_cest.date() + timedelta(days=1)
                next_day_9am_cest = cest_tz.localize(
                    datetime.combine(next_day, datetime.strptime("09:00:00", "%H:%M:%S").time())
                )
                # Convert back to UTC
                current_time = pd.Timestamp(next_day_9am_cest.astimezone(pytz.UTC)).tz_localize(None)
                logger.debug(f"Moving to next day: {current_time} UTC (09:00 CEST)")

        df = pd.DataFrame(future_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Apply same preprocessing - PASS THE ATTRACTION NAME!
        processed_df = self.preprocess_for_prediction(df, attraction_name)

        logger.info(f"Generated {len(processed_df)} future data points starting from {start_timestamp}")
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
        future_df = data_fetcher.generate_future_data(start_timestamp, prediction_steps, attraction_name) 
        
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
        
# Add this new endpoint to your main.py in the Cloud Run service

@app.route('/predictions/fast/<attraction_id>', methods=['GET'])
def get_predictions_fast(attraction_id: str):
    """
    Fast retrieval of pre-computed predictions for a specific attraction and date.
    This reads from the predictedQueueTimes collection instead of computing predictions.
    
    Usage: GET /predictions/fast/silver-star?date=2025-06-09
    """
    try:
        date_str = request.args.get('date')  # Format: YYYY-MM-DD
        hours = request.args.get('hours', 24)  # How many hours of predictions to return
        
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
        
        logger.info(f"Fast retrieval for {attraction_name} on {date_str}")
        
        # Get predictions from Firestore
        predictions = get_saved_predictions(attraction_name, target_date, int(hours))
        
        if not predictions:
            return jsonify({
                "attraction_id": attraction_id,
                "attraction_name": attraction_name,
                "date": date_str,
                "predictions": [],
                "message": "No predictions found for this date. Run batch predictions first.",
                "cache_miss": True
            })
        
        response = {
            "attraction_id": attraction_id,
            "attraction_name": attraction_name,
            "date": date_str,
            "predictions_count": len(predictions),
            "predictions": predictions,
            "cache_hit": True,
            "retrieved_at": datetime.now().isoformat()
        }
        
        logger.info(f"Retrieved {len(predictions)} cached predictions for {attraction_name}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error retrieving fast predictions: {str(e)}", exc_info=True)
        return jsonify({"error": f"Failed to retrieve predictions: {str(e)}"}), 500

def get_saved_predictions(attraction_name: str, date: datetime.date, hours: int = 24) -> List[Dict]:
    """
    Retrieve saved predictions from the predictedQueueTimes collection.
    """
    try:
        # Calculate time range
        start_time = datetime.combine(date, datetime.min.time())
        end_time = start_time + timedelta(hours=hours)
        
        # Format for document ID filtering
        start_doc_id = start_time.strftime("%Y%m%d%H%M%S")
        end_doc_id = end_time.strftime("%Y%m%d%H%M%S")
        
        logger.info(f"Querying predictions from {start_doc_id} to {end_doc_id}")
        
        # Get predictions collection
        attraction_ref = db.collection("attractions").document(attraction_name)
        predictions_ref = attraction_ref.collection("predictedQueueTimes")
        
        # Query for predictions in the date range
        query = predictions_ref.order_by("__name__").start_at([start_doc_id]).end_before([end_doc_id])
        docs = list(query.stream())
        
        logger.info(f"Found {len(docs)} prediction documents")
        
        predictions = []
        for doc in docs:
            doc_data = doc.to_dict()
            
            # Convert Firestore types to Python types
            prediction = {
                "timestamp": doc_data["timestamp"].isoformat() if hasattr(doc_data["timestamp"], 'isoformat') else str(doc_data["timestamp"]),
                "predicted_wait_time": float(doc_data.get("predicted_wait_time", 0)),
                "baseline_prediction": float(doc_data.get("baseline_prediction", 0)),
                "residual_prediction": float(doc_data.get("residual_prediction", 0)),
                "temperature": float(doc_data.get("temperature", 0)) if doc_data.get("temperature") is not None else None,
                "rain": float(doc_data.get("rain", 0)) if doc_data.get("rain") is not None else None,
                "is_holiday": bool(doc_data.get("is_holiday", False)),
                "model_version": doc_data.get("model_version", "unknown"),
                "prediction_created_at": doc_data.get("prediction_created_at").isoformat() if doc_data.get("prediction_created_at") else None
            }
            
            predictions.append(prediction)
        
        # Sort by timestamp to ensure correct order
        predictions.sort(key=lambda x: x["timestamp"])
        
        return predictions
        
    except Exception as e:
        logger.error(f"Error getting saved predictions: {e}")
        return []

@app.route('/predictions/status', methods=['GET'])
def prediction_status():
    """
    Get status of prediction system - shows what predictions are available.
    """
    try:
        # Check which attractions have recent predictions
        status = {}
        
        # Get sample of attractions
        attractions_ref = db.collection("attractions")
        attractions = list(attractions_ref.limit(10).stream())
        
        for attraction_doc in attractions:
            attraction_name = attraction_doc.id
            
            try:
                # Check for recent predictions
                predictions_ref = attraction_doc.reference.collection("predictedQueueTimes")
                recent_docs = list(predictions_ref.order_by("__name__", "DESCENDING").limit(5).stream())
                
                if recent_docs:
                    latest_doc = recent_docs[0]
                    latest_doc_data = latest_doc.to_dict()
                    
                    status[attraction_name] = {
                        "has_predictions": True,
                        "latest_prediction_time": latest_doc_data.get("timestamp").isoformat() if latest_doc_data.get("timestamp") else None,
                        "prediction_count": len(recent_docs),
                        "latest_created_at": latest_doc_data.get("prediction_created_at").isoformat() if latest_doc_data.get("prediction_created_at") else None
                    }
                else:
                    status[attraction_name] = {
                        "has_predictions": False,
                        "prediction_count": 0
                    }
                    
            except Exception as e:
                status[attraction_name] = {
                    "error": str(e)
                }
        
        return jsonify({
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "attractions": status,
            "message": "Use /predictions/fast/<attraction_id>?date=YYYY-MM-DD for fast retrieval"
        })
        
    except Exception as e:
        logger.error(f"Error getting prediction status: {e}")
        return jsonify({"error": str(e)}), 500
        
@app.route('/debug/attractions', methods=['GET'])
def debug_attractions():
    """Debug endpoint to check attraction matching between mappings and Firestore"""
    try:
        debug_info = {
            "timestamp": datetime.now().isoformat(),
            "mapping_count": len(LOCAL_RIDE_MAPPINGS),
            "mappings": [],
            "firestore_attractions": [],
            "matching_issues": []
        }
        
        # First, get actual Firestore attractions to populate the list properly
        try:
            attractions_ref = db.collection("attractions")
            firestore_docs = list(attractions_ref.limit(50).stream())
            
            for doc in firestore_docs:
                debug_info["firestore_attractions"].append(doc.id)
                
            logger.info(f"Found {len(firestore_docs)} attractions in Firestore")
                
        except Exception as e:
            debug_info["firestore_error"] = str(e)
            logger.error(f"Error listing Firestore attractions: {e}")
        
        # Check all mapped attractions
        for mapping in LOCAL_RIDE_MAPPINGS:
            attraction_name = mapping["name"]
            
            # Check if attraction exists in Firestore
            try:
                attraction_ref = db.collection("attractions").document(attraction_name)
                attraction_doc = attraction_ref.get()
                
                # FIXED: Proper document existence check
                exists_in_firestore = attraction_doc.exists
                
                logger.debug(f"Checking {attraction_name}: exists = {exists_in_firestore}")
                
                # Check for recent queue times - SIMPLE APPROACH (no ordering to avoid index requirement)
                queue_times_ref = attraction_ref.collection("queueTimes")
                
                # Just get a few documents without ordering (avoids index requirement)
                recent_docs = list(queue_times_ref.limit(5).stream())
                
                mapping_info = {
                    "id": mapping["id"],
                    "name": mapping["name"],
                    "normalized_name": mapping["normalized_name"],
                    "exists_in_firestore": exists_in_firestore,
                    "has_queue_times": len(recent_docs) > 0,
                    "recent_queue_count": len(recent_docs),
                    "sample_queue_doc_id": recent_docs[0].id if recent_docs else None
                }
                
                # Check for model files
                normalized_name = mapping["normalized_name"].lower()
                try:
                    storage_client = storage.Client()
                    bucket = storage_client.bucket(MODEL_BUCKET)
                    
                    gb_blob = bucket.blob(f"models/{normalized_name}_gb_baseline.pkl")
                    tcn_blob = bucket.blob(f"models/{normalized_name}_cached_scheduled_sampling_tcn.pt")
                    
                    mapping_info["has_gb_model"] = gb_blob.exists()
                    mapping_info["has_tcn_model"] = tcn_blob.exists()
                    mapping_info["has_both_models"] = gb_blob.exists() and tcn_blob.exists()
                    
                except Exception as e:
                    mapping_info["model_check_error"] = str(e)
                
                debug_info["mappings"].append(mapping_info)
                
                # Flag potential issues
                if not exists_in_firestore:
                    debug_info["matching_issues"].append(f"Attraction '{attraction_name}' not found in Firestore")
                elif not len(recent_docs) > 0:
                    debug_info["matching_issues"].append(f"Attraction '{attraction_name}' has no queue times data")
                elif not (mapping_info.get("has_both_models", False)):
                    debug_info["matching_issues"].append(f"Attraction '{attraction_name}' missing model files")
                    
            except Exception as e:
                debug_info["matching_issues"].append(f"Error checking '{attraction_name}': {str(e)}")
                logger.error(f"Error checking attraction {attraction_name}: {e}")
        
        # Summary
        total_mapped = len(LOCAL_RIDE_MAPPINGS)
        valid_attractions = len([m for m in debug_info["mappings"] if m.get("exists_in_firestore") and m.get("has_queue_times")])
        ready_for_prediction = len([m for m in debug_info["mappings"] if m.get("has_both_models", False)])
        
        debug_info["summary"] = {
            "total_mapped_attractions": total_mapped,
            "valid_in_firestore": valid_attractions,
            "ready_for_prediction": ready_for_prediction,
            "issues_count": len(debug_info["matching_issues"]),
            "firestore_attractions_found": len(debug_info["firestore_attractions"])
        }
        
        return jsonify(debug_info)
        
    except Exception as e:
        logger.error(f"Debug attractions error: {e}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/debug/simple-firestore', methods=['GET'])
def debug_simple_firestore():
    """Simplified debug to understand what's happening with Firestore"""
    try:
        debug_info = {
            "timestamp": datetime.now().isoformat(),
            "tests": []
        }
        
        # Test 1: Try to list attractions collection
        try:
            attractions_ref = db.collection("attractions")
            logger.info("Step 1: Got attractions collection reference")
            
            firestore_docs = list(attractions_ref.limit(10).stream())
            logger.info(f"Step 2: Listed {len(firestore_docs)} documents")
            
            attraction_names = [doc.id for doc in firestore_docs]
            
            debug_info["tests"].append({
                "test": "list_attractions_collection",
                "success": True,
                "result": f"Found {len(firestore_docs)} attractions",
                "attraction_names": attraction_names
            })
            
        except Exception as e:
            debug_info["tests"].append({
                "test": "list_attractions_collection", 
                "success": False,
                "error": str(e)
            })
            logger.error(f"Failed to list attractions: {e}")
        
        # Test 2: Try to get a specific attraction document (Silver Star)
        try:
            silver_star_ref = db.collection("attractions").document("Silver Star")
            silver_star_doc = silver_star_ref.get()
            
            debug_info["tests"].append({
                "test": "get_silver_star_document",
                "success": True,
                "exists": silver_star_doc.exists,
                "doc_id": silver_star_doc.id if silver_star_doc.exists else None,
                "has_data": bool(silver_star_doc.to_dict()) if silver_star_doc.exists else False
            })
            
        except Exception as e:
            debug_info["tests"].append({
                "test": "get_silver_star_document",
                "success": False, 
                "error": str(e)
            })
        
        # Test 3: Try to access queue times for Silver Star
        try:
            queue_ref = db.collection("attractions").document("Silver Star").collection("queueTimes")
            queue_docs = list(queue_ref.limit(3).stream())
            
            debug_info["tests"].append({
                "test": "get_silver_star_queue_times",
                "success": True,
                "queue_docs_found": len(queue_docs),
                "sample_doc_ids": [doc.id for doc in queue_docs]
            })
            
        except Exception as e:
            debug_info["tests"].append({
                "test": "get_silver_star_queue_times",
                "success": False,
                "error": str(e)
            })
        
        # Test 4: Check Firestore client status
        try:
            # Try a simple query to check if client is working
            test_query = db.collection("attractions").limit(1)
            test_docs = list(test_query.stream())
            
            debug_info["tests"].append({
                "test": "firestore_client_status",
                "success": True,
                "can_query": True,
                "test_docs_count": len(test_docs)
            })
            
        except Exception as e:
            debug_info["tests"].append({
                "test": "firestore_client_status",
                "success": False,
                "error": str(e)
            })
        
        # Test 5: Check if specific known documents exist
        known_attractions = ["Silver Star", "blue fire Megacoaster", "Voletarium"]
        for attraction in known_attractions:
            try:
                doc_ref = db.collection("attractions").document(attraction)
                doc = doc_ref.get()
                
                debug_info["tests"].append({
                    "test": f"check_specific_attraction_{attraction.replace(' ', '_')}",
                    "success": True,
                    "attraction_name": attraction,
                    "exists": doc.exists,
                    "document_id": doc.id
                })
                
            except Exception as e:
                debug_info["tests"].append({
                    "test": f"check_specific_attraction_{attraction.replace(' ', '_')}",
                    "success": False,
                    "attraction_name": attraction,
                    "error": str(e)
                })
        
        return jsonify(debug_info)
        
    except Exception as e:
        logger.error(f"Simple debug error: {e}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/debug/model-mappings', methods=['GET'])
def debug_model_mappings():
    """Debug endpoint to check model file mappings and availability"""
    try:
        debug_info = {
            "timestamp": datetime.now().isoformat(),
            "total_mapped_attractions": len(MODEL_FILE_MAPPINGS),
            "model_mappings": [],
            "storage_bucket": MODEL_BUCKET,
            "issues": []
        }
        
        # Check each mapping
        for firestore_name, model_file_name in MODEL_FILE_MAPPINGS.items():
            mapping_info = {
                "firestore_name": firestore_name,
                "model_file_name": model_file_name,
                "gb_file": f"models/{model_file_name}_gb_baseline.pkl",
                "tcn_file": f"models/{model_file_name}_cached_scheduled_sampling_tcn.pt"
            }
            
            # Check if files exist in storage
            try:
                bucket = storage_client.bucket(MODEL_BUCKET)
                
                gb_blob = bucket.blob(mapping_info["gb_file"])
                tcn_blob = bucket.blob(mapping_info["tcn_file"])
                
                mapping_info["gb_exists"] = gb_blob.exists()
                mapping_info["tcn_exists"] = tcn_blob.exists()
                mapping_info["both_exist"] = gb_blob.exists() and tcn_blob.exists()
                
                if not mapping_info["both_exist"]:
                    issue = f"Missing files for {firestore_name}: "
                    if not mapping_info["gb_exists"]:
                        issue += f"GB model missing ({mapping_info['gb_file']}) "
                    if not mapping_info["tcn_exists"]:
                        issue += f"TCN model missing ({mapping_info['tcn_file']}) "
                    debug_info["issues"].append(issue)
                
            except Exception as e:
                mapping_info["error"] = str(e)
                debug_info["issues"].append(f"Error checking {firestore_name}: {str(e)}")
            
            debug_info["model_mappings"].append(mapping_info)
        
        # Summary
        existing_models = sum(1 for m in debug_info["model_mappings"] if m.get("both_exist", False))
        debug_info["summary"] = {
            "total_attractions": len(MODEL_FILE_MAPPINGS),
            "attractions_with_models": existing_models,
            "attractions_missing_models": len(MODEL_FILE_MAPPINGS) - existing_models,
            "issues_count": len(debug_info["issues"])
        }
        
        return jsonify(debug_info)
        
    except Exception as e:
        logger.error(f"Error in debug model mappings: {e}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/debug/model-file-mappings', methods=['GET'])  # RENAMED from debug/model-mappings
def debug_model_file_mappings():  # RENAMED function name
    """Debug endpoint to check model file mappings and availability"""
    try:
        debug_info = {
            "timestamp": datetime.now().isoformat(),
            "total_mapped_attractions": len(MODEL_FILE_MAPPINGS),
            "model_mappings": [],
            "storage_bucket": MODEL_BUCKET,
            "issues": []
        }
        
        # Check each mapping
        for firestore_name, model_file_name in MODEL_FILE_MAPPINGS.items():
            mapping_info = {
                "firestore_name": firestore_name,
                "model_file_name": model_file_name,
                "gb_file": f"models/{model_file_name}_gb_baseline.pkl",
                "tcn_file": f"models/{model_file_name}_cached_scheduled_sampling_tcn.pt"
            }
            
            # Check if files exist in storage
            try:
                bucket = storage_client.bucket(MODEL_BUCKET)
                
                gb_blob = bucket.blob(mapping_info["gb_file"])
                tcn_blob = bucket.blob(mapping_info["tcn_file"])
                
                mapping_info["gb_exists"] = gb_blob.exists()
                mapping_info["tcn_exists"] = tcn_blob.exists()
                mapping_info["both_exist"] = gb_blob.exists() and tcn_blob.exists()
                
                if not mapping_info["both_exist"]:
                    issue = f"Missing files for {firestore_name}: "
                    if not mapping_info["gb_exists"]:
                        issue += f"GB model missing "
                    if not mapping_info["tcn_exists"]:
                        issue += f"TCN model missing "
                    debug_info["issues"].append(issue)
                
            except Exception as e:
                mapping_info["error"] = str(e)
                debug_info["issues"].append(f"Error checking {firestore_name}: {str(e)}")
            
            debug_info["model_mappings"].append(mapping_info)
        
        # Summary
        existing_models = sum(1 for m in debug_info["model_mappings"] if m.get("both_exist", False))
        debug_info["summary"] = {
            "total_attractions": len(MODEL_FILE_MAPPINGS),
            "attractions_with_models": existing_models,
            "attractions_missing_models": len(MODEL_FILE_MAPPINGS) - existing_models,
            "issues_count": len(debug_info["issues"])
        }
        
        return jsonify(debug_info)
        
    except Exception as e:
        logger.error(f"Error in debug model mappings: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/debug/time-windows', methods=['GET'])
def debug_time_windows():
    """Debug time window generation for 09:00-17:30 CEST - FIXED interval counting"""
    try:
        # Test time window generation
        import pytz
        cest_tz = pytz.timezone('Europe/Zurich')
        
        # Get tomorrow in CEST timezone
        tomorrow_utc = datetime.now(pytz.UTC) + timedelta(days=1)
        tomorrow_cest = tomorrow_utc.astimezone(cest_tz)
        
        # Set to 9 AM CEST time
        start_time_cest = tomorrow_cest.replace(hour=9, minute=0, second=0, microsecond=0)
        
        # Convert back to UTC for consistent handling
        start_timestamp = start_time_cest.astimezone(pytz.UTC)
        start_timestamp = pd.Timestamp(start_timestamp).tz_localize(None)
        
        # Generate test time windows - SIMULATE THE REAL LOGIC
        current_time = start_timestamp
        test_times = []
        steps_generated = 0
        max_operating_steps = 34  # Test 2 full days (17 × 2)
        safety_counter = 0
        max_safety_counter = 100  # Prevent infinite loops
        
        while steps_generated < max_operating_steps and safety_counter < max_safety_counter:
            safety_counter += 1
            
            # Convert to CEST for hour checking
            current_time_cest = current_time.tz_localize('UTC').astimezone(cest_tz)
            hour = current_time_cest.hour
            minute = current_time_cest.minute
            
            # ENFORCED: Operating hours 09:00 to 17:30 CEST
            is_operating_hours = (hour >= 9 and hour < 17) or (hour == 17 and minute <= 30)
            
            test_times.append({
                "timestamp_utc": current_time.isoformat(),
                "timestamp_cest": current_time_cest.strftime('%Y-%m-%d %H:%M:%S %Z'),
                "hour_cest": hour,
                "minute_cest": minute,
                "is_operating_hours": is_operating_hours,
                "step": steps_generated if is_operating_hours else "skipped",
                "safety_counter": safety_counter
            })
            
            if is_operating_hours:
                steps_generated += 1
            
            # Move to next 30-minute interval
            current_time += pd.Timedelta(minutes=30)
            
            # CRITICAL: Skip to next day if past operating hours (17:30 CEST)
            current_time_cest_check = current_time.tz_localize('UTC').astimezone(cest_tz)
            hour_check = current_time_cest_check.hour
            minute_check = current_time_cest_check.minute
            
            # If we just moved past 17:30, jump to next day at 09:00
            if hour_check >= 18 or (hour_check == 17 and minute_check > 30):
                # Move to next day at 09:00 CEST
                next_day = current_time_cest_check.date() + timedelta(days=1)
                next_day_9am_cest = cest_tz.localize(
                    datetime.combine(next_day, datetime.strptime("09:00:00", "%H:%M:%S").time())
                )
                # Convert back to UTC
                current_time = pd.Timestamp(next_day_9am_cest.astimezone(pytz.UTC)).tz_localize(None)
                
                # Add a marker for the jump
                test_times.append({
                    "timestamp_utc": current_time.isoformat(),
                    "timestamp_cest": f"JUMPED TO NEXT DAY: {current_time.tz_localize('UTC').astimezone(cest_tz).strftime('%Y-%m-%d %H:%M:%S %Z')}",
                    "hour_cest": 9,
                    "minute_cest": 0,
                    "is_operating_hours": True,
                    "step": "DAY_JUMP",
                    "safety_counter": safety_counter
                })
        
        # Calculate actual intervals per day from the data
        operating_intervals = [t for t in test_times if t["is_operating_hours"] and isinstance(t["step"], int)]
        
        # Group by date to count intervals per day
        intervals_by_date = {}
        for interval in operating_intervals:
            timestamp_cest = pd.to_datetime(interval["timestamp_cest"])
            date_key = timestamp_cest.date()
            if date_key not in intervals_by_date:
                intervals_by_date[date_key] = []
            intervals_by_date[date_key].append(interval)
        
        # Calculate the intervals per day
        intervals_per_day_actual = [len(intervals) for intervals in intervals_by_date.values()]
        
        return jsonify({
            "timezone": "Central European Summer Time (CEST, GMT+2)",
            "start_time_cest": start_time_cest.isoformat(),
            "start_timestamp_utc": start_timestamp.isoformat(),
            "operating_hours_logic": "09:00 to 17:30 CEST",
            "expected_intervals_per_day": 17,
            "actual_intervals_per_day": intervals_per_day_actual,
            "test_intervals": test_times,
            "operating_intervals_count": len(operating_intervals),
            "total_intervals_tested": len(test_times),
            "days_tested": len(intervals_by_date),
            "sample_operating_times_cest": [
                t["timestamp_cest"] for t in operating_intervals[:10]
            ],
            "intervals_by_date": {
                str(date): len(intervals) for date, intervals in intervals_by_date.items()
            },
            "validation": {
                "expected_per_day": 17,
                "actual_per_day": intervals_per_day_actual,
                "is_correct": all(count == 17 for count in intervals_per_day_actual),
                "safety_counter_used": safety_counter
            }
        })
        
    except Exception as e:
        logger.error(f"Error in debug time windows: {e}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/debug/features/<attraction_name>', methods=['GET'])
def debug_features(attraction_name: str):
    """Debug feature creation for a specific attraction"""
    try:
        # Create test data
        test_data = pd.DataFrame({
            'timestamp': [pd.Timestamp("2025-06-10 15:30:00")],
            'wait_time': [15.0],
            'closed': [0],
            'temperature': [20.0],
            'rain': [0.0],
            'temperature_unscaled': [20.0],
            'rain_unscaled': [0.0],
            'is_german_holiday': [0],
            'is_swiss_holiday': [0], 
            'is_french_holiday': [0]
        })
        
        # Process features
        processed_df = data_fetcher.preprocess_for_prediction(test_data, attraction_name)
        
        # Get feature names
        feature_cols = [col for col in processed_df.columns if col not in ['wait_time', 'timestamp']]
        
        return jsonify({
            "attraction_name": attraction_name,
            "feature_count": len(feature_cols),
            "features": feature_cols,
            "sample_values": {col: float(processed_df[col].iloc[0]) for col in feature_cols[:20]},
            "all_features": feature_cols
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/predict-single-day', methods=['POST'])
def predict_single_day():
    """
    Predict wait times for a single day for all attractions.
    This endpoint supports incremental prediction where previous predictions
    can be used as historical data for subsequent days.
    """
    try:
        data = request.get_json() or {}
        target_date = data.get('target_date')  # Format: YYYY-MM-DD
        previous_predictions = data.get('previous_predictions', {})  # Dict of attraction -> predictions
        trigger_source = data.get('trigger_source', 'manual')
        
        if not target_date:
            return jsonify({"error": "target_date is required (format: YYYY-MM-DD)"}), 400
        
        # Parse target date
        try:
            target_date_obj = datetime.strptime(target_date, "%Y-%m-%d").date()
        except ValueError:
            return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400
        
        # Validate it's a future date
        if target_date_obj <= datetime.now().date():
            return jsonify({"error": "target_date must be a future date"}), 400
        
        logger.info(f"Starting single day prediction for {target_date} (triggered by: {trigger_source})")
        
        # Import here to avoid circular imports
        from single_day_predictor import SingleDayPredictor
        
        predictor = SingleDayPredictor()
        results = predictor.predict_single_day_all_attractions(
            target_date_obj, 
            previous_predictions
        )
        
        # Count successful predictions
        successful = sum(1 for r in results.values() if r["status"] == "success")
        total = len(results)
        
        response = {
            "status": "completed",
            "target_date": target_date,
            "trigger_source": trigger_source,
            "total_attractions": total,
            "successful_attractions": successful,
            "failed_attractions": total - successful,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Single day prediction completed: {successful}/{total} attractions successful for {target_date}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Single day prediction error: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500
        
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)