import logging
import pandas as pd
import numpy as np
from typing import Tuple, List

logger = logging.getLogger(__name__)


def preprocess_data(df: pd.DataFrame, target_ride: str) -> pd.DataFrame:
    """Preprocess the data for autoregressive training"""
    logger.info(f"Preprocessing data for ride: {target_ride}")
    
    # Remove time_bucket if present
    if 'time_bucket' in df.columns:
        df = df.drop(columns=['time_bucket'])
    
    # Filter for target ride
    ride_col = f'ride_name_{target_ride}'
    if ride_col in df.columns:
        df = df[df[ride_col] == 1].copy()
        logger.info(f"Filtered data for ride {target_ride}: {len(df)} samples")
    
    # Remove all ride_name columns
    ride_cols = [col for col in df.columns if col.startswith('ride_name_')]
    df = df.drop(columns=ride_cols)
    
    # Handle missing values
    df = df.fillna(0)
    
    return df


def create_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Create static feature columns (excluding wait_time and timestamp)"""
    # Static features are everything except wait_time and timestamp
    # wait_time will be used autoregressively, not as a static feature
    static_feature_cols = [col for col in df.columns 
                          if col not in ['wait_time', 'timestamp']]
    
    logger.info(f"Static features: {static_feature_cols}")
    logger.info(f"Total static features: {len(static_feature_cols)}")
    
    return df, static_feature_cols


def load_data_splits(splits_output_dir: str, target_ride: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load train/val/test split indices"""
    train_indices = pd.read_parquet(f"{splits_output_dir}/train_indices.parquet")
    val_indices = pd.read_parquet(f"{splits_output_dir}/validation_indices.parquet")
    test_indices = pd.read_parquet(f"{splits_output_dir}/test_indices.parquet")
    
    ride_name_normalized = target_ride.replace(' ', '_')
    train_idx = train_indices[train_indices['ride_name'] == ride_name_normalized]['original_index'].values
    val_idx = val_indices[val_indices['ride_name'] == ride_name_normalized]['original_index'].values
    test_idx = test_indices[test_indices['ride_name'] == ride_name_normalized]['original_index'].values
    
    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError(f"No indices found for ride {target_ride}")
    
    logger.info(f"Data splits - Train: {len(train_idx)}, Validation: {len(val_idx)}, Test: {len(test_idx)}")
    return train_idx, val_idx, test_idx


def prepare_data_for_training(config: dict) -> dict:
    """Prepare all data needed for training"""
    # Load and preprocess data
    df = pd.read_parquet(config['data_path'])
    df = preprocess_data(df, config['target_ride'])
    
    # Load splits
    train_indices, val_indices, test_indices = load_data_splits(
        config['splits_output_dir'], 
        config['target_ride']
    )
    
    # Create features
    df, static_feature_cols = create_features(df)
    
    # Split data
    train_df = df.iloc[train_indices].copy()
    val_df = df.iloc[val_indices].copy()
    test_df = df.iloc[test_indices].copy()
    
    # Prepare features and targets
    X_train_static = train_df[static_feature_cols].values
    y_train = train_df['wait_time'].values
    X_val_static = val_df[static_feature_cols].values
    y_val = val_df['wait_time'].values
    X_test_static = test_df[static_feature_cols].values
    y_test = test_df['wait_time'].values
    
    return {
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df,
        'X_train_static': X_train_static,
        'y_train': y_train,
        'X_val_static': X_val_static,
        'y_val': y_val,
        'X_test_static': X_test_static,
        'y_test': y_test,
        'static_feature_cols': static_feature_cols,
        'train_indices': train_indices,
        'val_indices': val_indices,
        'test_indices': test_indices
    }


def calculate_scheduled_sampling_probability(current_epoch: int, total_epochs: int, 
                                            sampling_strategy: str = "linear") -> float:
    """Calculate teacher forcing probability based on current epoch and strategy"""
    if total_epochs <= 1:
        return 1.0
    
    if sampling_strategy == "linear":
        # Linear decay: 2% per epoch from 1.0 to minimum 0.02
        decay_rate = 0.02  # 2% per epoch
        min_prob = 0.02    # Minimum teacher forcing probability
        prob = 1.0 - (decay_rate * current_epoch)
        return max(min_prob, prob)
    elif sampling_strategy == "exponential":
        # Exponential decay: faster initial drop
        decay_factor = 0.95  # 5% decay per epoch
        min_prob = 0.05
        prob = 1.0 * (decay_factor ** current_epoch)
        return max(min_prob, prob)
    elif sampling_strategy == "inverse_sigmoid":
        # Sigmoid-based: steeper transition around epoch 15-20
        midpoint = total_epochs * 0.4  # Transition around 40% of training
        steepness = 0.3  # Controls how steep the transition is
        progress = (current_epoch - midpoint) * steepness
        prob = 1.0 / (1.0 + np.exp(progress))
        return max(0.05, prob)
    else:
        # Default to linear with 3% decay
        decay_rate = 0.03
        min_prob = 0.05
        prob = 1.0 - (decay_rate * current_epoch)
        return max(min_prob, prob)