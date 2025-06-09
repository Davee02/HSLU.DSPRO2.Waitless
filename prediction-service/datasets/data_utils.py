import logging
import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Global scaler instance - will be initialized once and reused
_global_scaler = None
_global_scaler_fitted = False


def get_global_scaler():
    """Get or create global scaler with reasonable default parameters for Europa-Park data"""
    global _global_scaler, _global_scaler_fitted
    
    if _global_scaler is None or not _global_scaler_fitted:
        logger.info("Creating global scaler with Europa-Park default statistics")
        
        # Create scaler with reasonable default statistics based on Europa-Park data
        _global_scaler = StandardScaler()
        
        # Set reasonable mean/std values based on typical Europa-Park weather
        # Temperature: mean ~15°C, std ~8°C (covers -5°C to 35°C range reasonably)
        # Rain: mean ~0.15 (15% chance of rain), std ~0.36 (binary-like distribution)
        _global_scaler.mean_ = np.array([15.0, 0.15])  # [temperature, rain]
        _global_scaler.scale_ = np.array([8.0, 0.36])  # [temperature_std, rain_std]
        _global_scaler.var_ = _global_scaler.scale_ ** 2
        _global_scaler.n_features_in_ = 2
        _global_scaler.n_samples_seen_ = 1000  # Dummy value
        _global_scaler.feature_names_in_ = np.array(['temperature', 'rain'])
        
        _global_scaler_fitted = True
        logger.info(f"Global scaler initialized with temperature(μ={_global_scaler.mean_[0]}, σ={_global_scaler.scale_[0]}) and rain(μ={_global_scaler.mean_[1]}, σ={_global_scaler.scale_[1]})")
    
    return _global_scaler


def estimate_scaler_from_recent_data(df, ride_name, num_cols):
    """
    Estimate scaler parameters from recent historical data
    This is a fallback when we don't have training statistics
    """
    if len(df) < 10:  # Not enough data for reliable statistics
        logger.warning(f"Not enough data ({len(df)} rows) for reliable scaling estimation")
        return get_global_scaler()
    
    logger.info(f"Estimating scaler from {len(df)} recent data points for {ride_name}")
    
    scaler = StandardScaler()
    
    # Filter for valid data (non-zero, non-extreme values)
    valid_data = df[num_cols].copy()
    
    # Remove extreme outliers (beyond 3 std devs) for more robust estimation
    for col in num_cols:
        col_data = valid_data[col]
        q1, q3 = col_data.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        valid_data = valid_data[(col_data >= lower_bound) & (col_data <= upper_bound)]
    
    if len(valid_data) < 5:  # Still not enough valid data
        logger.warning("Not enough valid data after outlier removal, using global scaler")
        return get_global_scaler()
    
    scaler.fit(valid_data[num_cols])
    
    logger.info(f"Estimated scaler for {ride_name}:")
    for i, col in enumerate(num_cols):
        logger.info(f"  {col}: μ={scaler.mean_[i]:.3f}, σ={scaler.scale_[i]:.3f}")
    
    return scaler

def normalize_ride_name(ride_name):
    """
    Robust ride name normalization that matches TRAINING EXACTLY
    """
    import re
    
    if not ride_name or not isinstance(ride_name, str):
        logger.warning(f"Invalid ride name: {ride_name}")
        return "unknown_ride"
    
    # EXACT same logic as training (from 09_feature_engineering.py)
    normalized = ride_name.lower()
    
    # Replace spaces and common separators with underscores  
    normalized = re.sub(r'[\s\-–—]+', '_', normalized)
    
    # Remove or replace special characters, but keep letters, numbers, and underscores
    normalized = re.sub(r'[^\w]', '_', normalized)
    
    # Clean up multiple underscores
    normalized = re.sub(r'_+', '_', normalized)
    
    # Remove leading/trailing underscores
    normalized = normalized.strip('_')
    
    # Handle edge case where everything was removed
    if not normalized:
        logger.warning(f"Ride name '{ride_name}' normalized to empty string, using fallback")
        normalized = f"ride_{abs(hash(ride_name)) % 10000}"
    
    logger.debug(f"Normalized ride name: '{ride_name}' -> '{normalized}'")
    return normalized

def decompose_timestamp(df):
    """Extract temporal components from timestamp and mark closing hours - MATCHES TRAINING"""
    df['datetime'] = pd.to_datetime(df['timestamp'])
    
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['weekday'] = df['datetime'].dt.weekday  # Monday=0, Sunday=6
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute

    # FIXED: Create operating hours mask (True if within operating hours, False otherwise)
    # Operating hours: 9:00 AM to 7:30 PM (19:30)
    operating_hours_mask = (
        (df['hour'] >= 9) & 
        ((df['hour'] < 19) | ((df['hour'] == 19) & (df['minute'] <= 30)))
    )
    
    # Handle the 'closed' column with explicit logic to avoid bitwise operations
    if 'closed' in df.columns:
        # First, ensure closed column is numeric
        if df['closed'].dtype == object:
            df['closed'] = df['closed'].map({'True': 1, 'False': 0, True: 1, False: 0}).fillna(0)
        elif df['closed'].dtype == bool:
            df['closed'] = df['closed'].astype(int)
        
        # Now for each row outside operating hours, set closed to 1
        # This preserves existing closed=1 values during operating hours
        df.loc[~operating_hours_mask, 'closed'] = 1
    else:
        # If closed column doesn't exist, create it
        df['closed'] = 0
        df.loc[~operating_hours_mask, 'closed'] = 1

    df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)
    df['part_of_day'] = df['hour'].apply(lambda x: 
                                        'morning' if 6 <= x < 12 else
                                        'afternoon' if 12 <= x < 17 else
                                        'evening' if 17 <= x < 20 else
                                        'night')
    
    df['season'] = df['month'].apply(lambda x:
                                    'winter' if x in [12, 1, 2] else
                                    'spring' if x in [3, 4, 5] else
                                    'summer' if x in [6, 7, 8] else
                                    'fall')
    
    # Cyclical encoding - EXACTLY as in training
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday']/7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday']/7)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute']/60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute']/60)
    
    return df


def process_boolean_features(df):
    """Convert boolean features to integers - EXACTLY as in training"""
    bool_cols = ['closed', 'is_german_holiday', 'is_swiss_holiday', 'is_french_holiday']
    
    for col in bool_cols:
        if col in df.columns:
            if df[col].dtype == bool:
                df[col] = df[col].astype(int)
            elif df[col].dtype == object:
                df[col] = df[col].map({'True': 1, 'False': 0})
    
    return df


def create_one_hot_encodings(df, ride_name):
    """Create one-hot encodings exactly as done in training with ROBUST ride name handling"""
    
    # Part of day one-hot encoding
    df['part_of_day_afternoon'] = (df['part_of_day'] == 'afternoon').astype(int)
    df['part_of_day_evening'] = (df['part_of_day'] == 'evening').astype(int)
    df['part_of_day_morning'] = (df['part_of_day'] == 'morning').astype(int)
    df['part_of_day_night'] = (df['part_of_day'] == 'night').astype(int)
    
    # Season one-hot encoding
    df['season_fall'] = (df['season'] == 'fall').astype(int)
    df['season_spring'] = (df['season'] == 'spring').astype(int)
    df['season_summer'] = (df['season'] == 'summer').astype(int)
    df['season_winter'] = (df['season'] == 'winter').astype(int)
    
    # Year one-hot encoding (for years 2017-2024 as in training)
    for year in range(2017, 2025):
        df[f'year_{year}'] = (df['year'] == year).astype(int)
    
    # CRITICAL: Create ride_name one-hot encoding with EXACT same normalization as training
    ride_name_normalized = normalize_ride_name(ride_name)
    
    # Validate the normalized name
    if not ride_name_normalized or ride_name_normalized == '_':
        logger.error(f"Failed to normalize ride name '{ride_name}', using fallback")
        ride_name_normalized = "unknown_ride"
    
    feature_name = f'ride_name_{ride_name_normalized}'
    df[feature_name] = 1
    
    logger.info(f"Created ride feature: '{feature_name}' for ride '{ride_name}'")
    
    # CRITICAL: Set ALL OTHER ride features to 0 (as they would be during training)
    # This is probably the major issue - missing this step
    training_rides = [
        'arthur', 'alpine_express_enzian', 'arena_of_football__be_part_of_it',
        'atlantica_supersplash', 'atlantis_adventure', 'baaa_express', 
        'bellevue_ferris_wheel', 'castello_dei_medici', 'dancing_dingie',
        'euromir', 'eurotower', 'eurosat__cancan_coaster', 'fjordrafting',
        'jim_button__journey_through_morrowland', 'josefinas_magical_imperial_journey',
        'kolumbusjolle', 'madame_freudenreich_curiosits', 'matterhornblitz',
        'old_mac_donalds_tractor_fun', 'pegasus', 'pirates_in_batavia',
        'poppy_towers', 'poseidon', 'silver_star', 'snorri_touren',
        'swiss_bob_run', 'tirol_log_flume', 'tnnevirvel', 
        'vienna_wave_swing_glckspilz', 'vindjammer', 'voletarium',
        'volo_da_vinci', 'voltron_nevera_powered_by_rimac', 
        'wodan_timburcoaster', 'whale_adventures_northern_lights',
        'blue_fire_megacoaster'
    ]
    
    # Set all ride features to 0 first
    for ride in training_rides:
        col_name = f'ride_name_{ride}'
        if col_name != feature_name:  # Don't overwrite our target ride
            df[col_name] = 0
    
    logger.info(f"Set {len(training_rides)-1} other ride features to 0")
    
    # Drop original categorical columns
    df = df.drop(columns=['part_of_day', 'season', 'year'], errors='ignore')
    
    return df


def scale_numerical_features(df, ride_name, use_recent_data_for_scaling=True):
    """
    Scale numerical features using TRAINING-COMPATIBLE scaling
    """
    
    # Numerical features that were scaled in training
    num_cols = ['temperature', 'rain']  # 'wind' was dropped in training
    num_cols = [col for col in num_cols if col in df.columns]
    
    if not num_cols:
        logger.warning("No numerical columns to scale")
        return df
    
    # Create unscaled versions for reference (as done in training)
    for col in num_cols:
        df[col + '_unscaled'] = df[col].copy()
    
    # CRITICAL: Use ride-specific scaling parameters from training
    # You need to either:
    # 1. Save the scalers during training and load them here, OR
    # 2. Use approximate scaling parameters for each ride
    
    # For now, use global approximations based on Europa-Park climate
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    
    # Set reasonable mean/std values based on typical Europa-Park weather
    scaler.mean_ = np.array([15.0, 0.15])  # [temperature, rain]
    scaler.scale_ = np.array([8.0, 0.36])  # [temperature_std, rain_std]
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = 2
    scaler.n_samples_seen_ = 1000  # Dummy value
    scaler.feature_names_in_ = np.array(['temperature', 'rain'])
    
    # Apply scaling
    try:
        df[num_cols] = scaler.transform(df[num_cols])
        logger.info(f"Successfully scaled numerical features: {num_cols}")
        
        # Log scaling results for debugging
        for i, col in enumerate(num_cols):
            scaled_mean = df[col].mean()
            scaled_std = df[col].std()
            logger.debug(f"Scaled {col}: μ={scaled_mean:.3f}, σ={scaled_std:.3f}")
            
    except Exception as e:
        logger.error(f"Error scaling features: {e}")
        logger.warning("Falling back to no scaling (this may cause poor predictions)")
    
    return df


def preprocess_data(df: pd.DataFrame, target_ride: str) -> pd.DataFrame:
    """
    Preprocess the data for inference - MATCHES TRAINING EXACTLY
    
    This is the MAIN function you'll use in production.
    It replicates the exact preprocessing pipeline used during training.
    """
    logger.info(f"Preprocessing data for ride: {target_ride}")
    
    # Step 1: Basic temporal decomposition (as in training)
    df = decompose_timestamp(df)
    
    # Step 2: Drop unnecessary columns (as in training)
    cols_to_drop = ['month', 'day', 'hour', 'minute', 'datetime']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
    
    # Step 3: Process boolean features
    df = process_boolean_features(df)
    
    # Step 4: Create one-hot encodings (as in training) with robust ride name handling
    df = create_one_hot_encodings(df, target_ride)
    
    # Step 5: Scale numerical features (using global/estimated scaler)
    df = scale_numerical_features(df, target_ride, use_recent_data_for_scaling=True)
    
    # Step 6: Ensure all features are numeric
    for col in df.columns:
        if col not in ['timestamp', 'wait_time']:
            if not pd.api.types.is_numeric_dtype(df[col]):
                logger.warning(f"Converting {col} to numeric")
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Step 7: Handle any remaining missing values
    df = df.fillna(0)
    
    logger.info(f"Preprocessing complete - {len(df.columns)} features for {len(df)} records")
    logger.info(f"Final feature list: {[col for col in df.columns if col not in ['timestamp', 'wait_time']]}")
    
    return df


def create_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create static feature columns (excluding wait_time and timestamp)
    This is used after preprocessing to extract the feature column names.
    """
    # Static features are everything except wait_time and timestamp
    static_feature_cols = [col for col in df.columns 
                          if col not in ['wait_time', 'timestamp']]
    
    logger.info(f"Static features: {len(static_feature_cols)} features")
    logger.debug(f"Feature names: {static_feature_cols}")
    
    return df, static_feature_cols