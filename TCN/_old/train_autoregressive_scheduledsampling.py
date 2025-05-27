#!/usr/bin/env python3
"""
Autoregressive TCN Training Script with Cached Scheduled Sampling
Hybrid approach: GradientBoosting baseline + Autoregressive TCN for residuals
Implements cached scheduled sampling for performance and compatibility with torch.compile
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
import yaml
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import wandb
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import pickle
from pytorch_tcn import TCN

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cached_scheduled_sampling_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


class CachedScheduledSampling:
    """
    Manages cached predictions for scheduled sampling to avoid dynamic shapes
    and enable torch.compile compatibility.
    """
    def __init__(self, cache_update_frequency=5, max_cache_size=100000):
        """
        Args:
            cache_update_frequency: Update cache every N epochs
            max_cache_size: Maximum number of cached predictions
        """
        self.prediction_cache = {}  # {sample_idx: {timestep: (wait_pred, residual_pred)}}
        self.cache_epoch = -1
        self.cache_update_frequency = cache_update_frequency
        self.max_cache_size = max_cache_size
        logger.info(f"Initialized cached scheduled sampling (update every {cache_update_frequency} epochs)")
    
    def should_update_cache(self, current_epoch):
        """Check if cache should be updated"""
        # Update cache at epoch 0 and then every cache_update_frequency epochs
        if current_epoch == 0:
            return True
        elif current_epoch > 0 and current_epoch % self.cache_update_frequency == 0:
            return True
        else:
            return False
    
    def update_cache(self, model, gb_model, dataset, device, static_feature_cols):
        """
        Pre-compute model predictions for all relevant sample-timestep combinations.
        This runs in batches to avoid memory issues.
        """
        if model is None or gb_model is None:
            logger.warning("Models not available for cache update")
            return
        
        logger.info(f"Updating prediction cache for epoch {dataset.current_epoch}...")
        
        # Clear old cache
        self.prediction_cache.clear()
        
        # Set model to evaluation mode
        model_was_training = model.training
        model.eval()
        
        batch_size = 256  # Smaller batch size for cache update
        cache_count = 0
        
        try:
            with torch.no_grad():
                # Process samples in batches for memory efficiency
                for start_idx in range(0, len(dataset.valid_indices), batch_size):
                    end_idx = min(start_idx + batch_size, len(dataset.valid_indices))
                    batch_indices = dataset.valid_indices[start_idx:end_idx]
                    
                    for sample_idx in batch_indices:
                        if cache_count >= self.max_cache_size:
                            logger.warning(f"Cache size limit reached ({self.max_cache_size})")
                            break
                        
                        # Pre-compute predictions for this sample's autoregressive sequence
                        self._cache_sample_predictions(
                            sample_idx, model, gb_model, dataset, device, static_feature_cols
                        )
                        cache_count += 1
                    
                    if cache_count >= self.max_cache_size:
                        break
        
        except Exception as e:
            logger.error(f"Error updating cache: {e}")
        
        finally:
            # Restore model training state
            if model_was_training:
                model.train()
        
        # Update cache epoch to current epoch
        self.cache_epoch = dataset.current_epoch
        logger.info(f"Cache updated with {len(self.prediction_cache)} samples for epoch {self.cache_epoch}")
        logger.info(f"Next cache update scheduled for epoch {self.cache_epoch + self.cache_update_frequency}")
    
    
    def _cache_sample_predictions(self, target_idx, model, gb_model, dataset, device, static_feature_cols):
        """Cache predictions for a single sample's autoregressive sequence"""
        target_timestamp = dataset.timestamps.iloc[target_idx]
        target_date = dataset.dates.iloc[target_idx]
        
        sequence_start = target_idx - dataset.seq_length
        sequence_indices = list(range(sequence_start, target_idx))
        
        static_features = dataset.X_static[target_idx]
        sample_cache = {}
        
        # Build sequence incrementally and cache predictions at each step
        current_sequence = []
        
        for seq_position, seq_idx in enumerate(sequence_indices):
            seq_timestamp = dataset.timestamps.iloc[seq_idx]
            seq_date = dataset.dates.iloc[seq_idx]
            
            # Only cache predictions for same-day future predictions
            if (seq_date == target_date and 
                seq_timestamp.hour >= dataset.opening_hour):
                
                try:
                    # Predict based on current sequence state
                    pred_wait_time, pred_residual = self._generate_model_prediction(
                        static_features, current_sequence, seq_idx, model, gb_model, dataset
                    )
                    
                    # Cache the prediction
                    sample_cache[seq_position] = (pred_wait_time, pred_residual)
                    
                    # Add prediction to sequence for next iteration
                    current_sequence.extend([pred_wait_time, pred_residual])
                    
                except Exception as e:
                    logger.debug(f"Failed to cache prediction for sample {target_idx}, seq {seq_position}: {e}")
                    # Add actual values as fallback
                    actual_wait_time = dataset.wait_times[seq_idx]
                    actual_residual = dataset.residuals[seq_idx]
                    current_sequence.extend([actual_wait_time, actual_residual])
            else:
                # Add actual historical values (not same day)
                actual_wait_time = dataset.wait_times[seq_idx]
                actual_residual = dataset.residuals[seq_idx]
                current_sequence.extend([actual_wait_time, actual_residual])
        
        # Store cache for this sample
        if sample_cache:
            self.prediction_cache[target_idx] = sample_cache
    
    def _generate_model_prediction(self, static_features, sequence_features, target_idx, model, gb_model, dataset):
        """Generate model prediction for caching"""
        try:
            # Prepare sequence for model input
            expected_sequence_size = dataset.seq_length * 2
            current_sequence = np.array(sequence_features, dtype=np.float32)
            
            if len(current_sequence) < expected_sequence_size:
                # Pad with zeros if sequence too short
                padding_needed = expected_sequence_size - len(current_sequence)
                current_sequence = np.concatenate([
                    np.zeros(padding_needed, dtype=np.float32), 
                    current_sequence
                ])
            elif len(current_sequence) > expected_sequence_size:
                # Truncate if too long
                current_sequence = current_sequence[-expected_sequence_size:]
            
            # Combine features for model input
            combined_features = np.concatenate([static_features, current_sequence])
            model_input = torch.FloatTensor(combined_features).unsqueeze(0)
            
            # Get residual prediction
            residual_pred = model(model_input).cpu().numpy().flatten()[0]
            
            # Get baseline prediction
            gb_pred = gb_model.predict(static_features.reshape(1, -1))[0]
            
            # Combined prediction
            combined_pred = gb_pred + residual_pred
            
            # Add small amount of noise for training diversity
            wait_noise_std = max(0.05, abs(combined_pred) * 0.1)
            residual_noise_std = max(0.025, abs(residual_pred) * 0.1)
            
            noisy_wait_pred = combined_pred + np.random.normal(0, wait_noise_std)
            noisy_residual_pred = residual_pred + np.random.normal(0, residual_noise_std)
            
            return max(0, noisy_wait_pred), noisy_residual_pred
            
        except Exception as e:
            logger.debug(f"Model prediction failed during caching: {e}")
            # Fallback to noisy actual values
            actual_wait_time = dataset.wait_times[target_idx]
            actual_residual = dataset.residuals[target_idx]
            
            wait_noise_std = max(0.1, actual_wait_time * 0.15)
            residual_noise_std = max(0.05, abs(actual_residual) * 0.2)
            
            simulated_wait_time = actual_wait_time + np.random.normal(0, wait_noise_std)
            simulated_residual = actual_residual + np.random.normal(0, residual_noise_std)
            
            return max(0, simulated_wait_time), simulated_residual
    
    def get_cached_prediction(self, sample_idx, seq_position):
        """Get cached prediction for a specific sample and sequence position"""
        if sample_idx in self.prediction_cache:
            if seq_position in self.prediction_cache[sample_idx]:
                return self.prediction_cache[sample_idx][seq_position]
        return None
    
    def clear_cache(self):
        """Clear the prediction cache"""
        self.prediction_cache.clear()
        logger.info("Prediction cache cleared")


class AutoregressiveResidualsDataset(Dataset):
    """
    Original dataset for autoregressive residual prediction (used for validation/test).
    """
    def __init__(self, X_static, residuals, wait_times, seq_length, timestamps, 
                 opening_hour=9, closing_hour=21):
        self.X_static = X_static  
        self.residuals = residuals  
        self.wait_times = wait_times  
        self.seq_length = seq_length
        
        if isinstance(timestamps, pd.Series):
            self.timestamps = pd.to_datetime(timestamps.reset_index(drop=True))
        else:
            self.timestamps = pd.to_datetime(timestamps)
        
        self.opening_hour = opening_hour
        self.closing_hour = closing_hour
        
        self.dates = self.timestamps.dt.date
        self.valid_indices = self._create_valid_indices()
        
        logger.info(f"Original autoregressive dataset: {len(self.valid_indices)} valid sequences")
    
    def _create_valid_indices(self):
        """Create indices where we can form valid sequences"""
        valid_indices = []
        
        for i in range(self.seq_length, len(self.X_static)):
            current_timestamp = self.timestamps.iloc[i]
            current_hour = current_timestamp.hour
            
            if self.opening_hour <= current_hour <= self.closing_hour:
                valid_indices.append(i)
        
        return valid_indices
    
    def _get_autoregressive_sequence(self, target_idx):
        """Create sequence with simple noise injection for same-day predictions"""
        target_timestamp = self.timestamps.iloc[target_idx]
        target_date = self.dates.iloc[target_idx]
        
        sequence_start = target_idx - self.seq_length
        sequence_indices = list(range(sequence_start, target_idx))
        
        static_features = self.X_static[target_idx]
        autoregressive_features = []
        
        for seq_idx in sequence_indices:
            seq_timestamp = self.timestamps.iloc[seq_idx]
            seq_date = self.dates.iloc[seq_idx]
            
            actual_wait_time = self.wait_times[seq_idx]
            actual_residual = self.residuals[seq_idx]
            
            if (seq_date == target_date and 
                seq_timestamp.hour >= self.opening_hour):
                
                wait_noise_std = max(0.1, actual_wait_time * 0.15)
                residual_noise_std = max(0.05, abs(actual_residual) * 0.2)
                
                simulated_wait_time = actual_wait_time + np.random.normal(0, wait_noise_std)
                simulated_residual = actual_residual + np.random.normal(0, residual_noise_std)
                simulated_wait_time = max(0, simulated_wait_time)
                
                autoregressive_features.extend([simulated_wait_time, simulated_residual])
            else:
                autoregressive_features.extend([actual_wait_time, actual_residual])
        
        combined_features = np.concatenate([
            static_features,
            autoregressive_features
        ])
        
        target_residual = self.residuals[target_idx]
        
        return combined_features.astype(np.float32), np.array([target_residual], dtype=np.float32)
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        target_idx = self.valid_indices[idx]
        X, y = self._get_autoregressive_sequence(target_idx)
        return torch.FloatTensor(X), torch.FloatTensor(y)


class CachedScheduledSamplingDataset(Dataset):
    """
    Dataset for autoregressive residual prediction with cached scheduled sampling.
    Uses pre-computed predictions to avoid dynamic shapes during training.
    """
    def __init__(self, X_static, residuals, wait_times, seq_length, timestamps, 
                 opening_hour=9, closing_hour=21, current_epoch=0, total_epochs=100,
                 sampling_strategy="linear", noise_factor=0.15, 
                 prediction_cache=None):
        """
        Args:
            prediction_cache: CachedScheduledSampling instance
        """
        self.X_static = X_static
        self.residuals = residuals
        self.wait_times = wait_times
        self.seq_length = seq_length
        self.current_epoch = current_epoch
        self.total_epochs = total_epochs
        self.sampling_strategy = sampling_strategy
        self.noise_factor = noise_factor
        self.prediction_cache = prediction_cache
        
        if isinstance(timestamps, pd.Series):
            self.timestamps = pd.to_datetime(timestamps.reset_index(drop=True))
        else:
            self.timestamps = pd.to_datetime(timestamps)
        
        self.opening_hour = opening_hour
        self.closing_hour = closing_hour
        
        # For same-day logic
        self.dates = self.timestamps.dt.date
        
        self.valid_indices = self._create_valid_indices()
        
        # Calculate scheduled sampling probability
        self.teacher_forcing_prob = self._calculate_teacher_forcing_prob()
        
        logger.info(f"Cached Scheduled Sampling Dataset: {len(self.valid_indices)} valid sequences")
        logger.info(f"Current epoch: {current_epoch}, Teacher forcing prob: {self.teacher_forcing_prob:.3f}")
    
    def update_epoch(self, epoch):
        """Update current epoch and recalculate teacher forcing probability"""
        self.current_epoch = epoch
        self.teacher_forcing_prob = self._calculate_teacher_forcing_prob()
        logger.info(f"Updated epoch to {epoch}, Teacher forcing prob: {self.teacher_forcing_prob:.3f}")
    
    def _calculate_teacher_forcing_prob(self):
        """Calculate teacher forcing probability based on current epoch and strategy"""
        if self.total_epochs <= 1:
            return 1.0
        
        # For faster decay (3% per epoch), calculate based on epochs not total progress
        if self.sampling_strategy == "linear":
            # Linear decay: 3% per epoch from 1.0 to minimum 0.05
            decay_rate = 0.03  # 3% per epoch
            min_prob = 0.05    # Minimum teacher forcing probability
            prob = 1.0 - (decay_rate * self.current_epoch)
            return max(min_prob, prob)
        elif self.sampling_strategy == "exponential":
            # Exponential decay: faster initial drop
            decay_factor = 0.95  # 5% decay per epoch
            min_prob = 0.05
            prob = 1.0 * (decay_factor ** self.current_epoch)
            return max(min_prob, prob)
        elif self.sampling_strategy == "inverse_sigmoid":
            # Sigmoid-based: steeper transition around epoch 15-20
            midpoint = self.total_epochs * 0.4  # Transition around 40% of training
            steepness = 0.3  # Controls how steep the transition is
            progress = (self.current_epoch - midpoint) * steepness
            prob = 1.0 / (1.0 + np.exp(progress))
            return max(0.05, prob)
        else:
            # Default to linear with 3% decay
            decay_rate = 0.03
            min_prob = 0.05
            prob = 1.0 - (decay_rate * self.current_epoch)
            return max(min_prob, prob)
    
    def _create_valid_indices(self):
        """Create indices where we can form valid sequences"""
        valid_indices = []
        
        for i in range(self.seq_length, len(self.X_static)):
            current_timestamp = self.timestamps.iloc[i]
            current_hour = current_timestamp.hour
            
            # Only predict during operating hours
            if self.opening_hour <= current_hour <= self.closing_hour:
                valid_indices.append(i)
        
        return valid_indices
    
    def _get_autoregressive_sequence(self, target_idx):
        """
        Create sequence with cached scheduled sampling.
        Uses pre-computed predictions for performance and compatibility.
        """
        target_timestamp = self.timestamps.iloc[target_idx]
        target_date = self.dates.iloc[target_idx]
        
        sequence_start = target_idx - self.seq_length
        sequence_indices = list(range(sequence_start, target_idx))
        
        static_features = self.X_static[target_idx]
        autoregressive_features = []
        
        # Track scheduled sampling usage for debugging
        teacher_forcing_count = 0
        scheduled_sampling_count = 0
        cached_prediction_count = 0
        
        for seq_position, seq_idx in enumerate(sequence_indices):
            seq_timestamp = self.timestamps.iloc[seq_idx]
            seq_date = self.dates.iloc[seq_idx]
            
            actual_wait_time = self.wait_times[seq_idx]
            actual_residual = self.residuals[seq_idx]
            
            # Same-day prediction logic with cached scheduled sampling
            if (seq_date == target_date and 
                seq_timestamp.hour >= self.opening_hour):
                
                # Probabilistic scheduled sampling decision
                use_teacher_forcing = np.random.random() < self.teacher_forcing_prob
                
                if use_teacher_forcing:
                    # Teacher Forcing: Use actual ground truth values
                    autoregressive_features.extend([actual_wait_time, actual_residual])
                    teacher_forcing_count += 1
                else:
                    # Scheduled Sampling: Use cached predictions
                    cached_pred = None
                    if self.prediction_cache:
                        cached_pred = self.prediction_cache.get_cached_prediction(target_idx, seq_position)
                    
                    if cached_pred is not None:
                        pred_wait_time, pred_residual = cached_pred
                        autoregressive_features.extend([pred_wait_time, pred_residual])
                        scheduled_sampling_count += 1
                        cached_prediction_count += 1
                    else:
                        # Fallback to noisy actual values if no cache
                        wait_noise_std = max(0.1, actual_wait_time * self.noise_factor)
                        residual_noise_std = max(0.05, abs(actual_residual) * 0.2)
                        
                        simulated_wait_time = actual_wait_time + np.random.normal(0, wait_noise_std)
                        simulated_residual = actual_residual + np.random.normal(0, residual_noise_std)
                        simulated_wait_time = max(0, simulated_wait_time)
                        
                        autoregressive_features.extend([simulated_wait_time, simulated_residual])
                        scheduled_sampling_count += 1
            else:
                # Use actual historical values (not same day)
                autoregressive_features.extend([actual_wait_time, actual_residual])
        
        # Store sampling stats for periodic logging (every 1000th sample)
        if hasattr(self, '_sampling_stats'):
            self._sampling_stats['teacher_forcing'] += teacher_forcing_count
            self._sampling_stats['scheduled_sampling'] += scheduled_sampling_count
            self._sampling_stats['cached_predictions'] += cached_prediction_count
            self._sampling_stats['total_samples'] += 1
        else:
            self._sampling_stats = {
                'teacher_forcing': teacher_forcing_count,
                'scheduled_sampling': scheduled_sampling_count,
                'cached_predictions': cached_prediction_count,
                'total_samples': 1
            }
        
        # Log sampling statistics every 1000 samples
        if self._sampling_stats['total_samples'] % 1000 == 0:
            tf_pct = (self._sampling_stats['teacher_forcing'] / 
                     (self._sampling_stats['teacher_forcing'] + self._sampling_stats['scheduled_sampling'] + 1e-6)) * 100
            ss_pct = (self._sampling_stats['scheduled_sampling'] / 
                     (self._sampling_stats['teacher_forcing'] + self._sampling_stats['scheduled_sampling'] + 1e-6)) * 100
            cache_usage = (self._sampling_stats['cached_predictions'] / 
                          (self._sampling_stats['scheduled_sampling'] + 1e-6)) * 100
            
            logger.info(f"Sampling Stats (last 1000): TF={tf_pct:.1f}%, SS={ss_pct:.1f}%, Cache Usage={cache_usage:.1f}%")
        
        # Combine static and autoregressive features
        combined_features = np.concatenate([
            static_features,
            autoregressive_features
        ])
        
        # Target is the actual residual at target_idx
        target_residual = self.residuals[target_idx]
        
        return combined_features.astype(np.float32), np.array([target_residual], dtype=np.float32)
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        target_idx = self.valid_indices[idx]
        X, y = self._get_autoregressive_sequence(target_idx)
        return torch.FloatTensor(X), torch.FloatTensor(y)


class AutoregressiveTCNModel(nn.Module):
    """
    TCN model for autoregressive residual prediction.
    Takes static features + sequence of previous wait_times/residuals.
    """
    def __init__(self, static_features_size, seq_length, output_size, 
                 num_channels, kernel_size, dropout=0.2, num_layers=8):
        super(AutoregressiveTCNModel, self).__init__()
        
        self.static_features_size = static_features_size
        self.seq_length = seq_length
        
        # The input to TCN will be: static features repeated + autoregressive sequence
        # Autoregressive sequence: seq_length steps of [wait_time, residual] = seq_length * 2
        self.autoregressive_size = seq_length * 2  # wait_time + residual per timestep
        self.total_input_size = static_features_size + self.autoregressive_size
        
        # TCN processes the combined features as a sequence
        # We'll reshape the input to treat it as a sequence
        self.tcn_input_size = static_features_size + 2  # static + [wait_time, residual] per step
        
        channels = [num_channels] * num_layers
        
        self.tcn = TCN(
            num_inputs=self.tcn_input_size,
            num_channels=channels,
            kernel_size=kernel_size,
            dropout=dropout,
            causal=True,
            use_skip_connections=True
        )
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(num_channels, num_channels // 2)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(num_channels // 2, output_size)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Split input into static and autoregressive parts
        static_features = x[:, :self.static_features_size]  # (batch_size, static_features)
        autoregressive_features = x[:, self.static_features_size:]  # (batch_size, seq_length * 2)
        
        # Reshape autoregressive features
        autoregressive_reshaped = autoregressive_features.view(
            batch_size, self.seq_length, 2
        )  # (batch_size, seq_length, 2)
        
        # Repeat static features for each timestep
        static_repeated = static_features.unsqueeze(1).repeat(
            1, self.seq_length, 1
        )  # (batch_size, seq_length, static_features_size)
        
        # Combine static and autoregressive features
        combined_sequence = torch.cat([
            static_repeated, autoregressive_reshaped
        ], dim=2)  # (batch_size, seq_length, static_features_size + 2)
        
        # TCN expects (batch_size, features, seq_length)
        tcn_input = combined_sequence.transpose(1, 2)
        
        # TCN forward pass
        tcn_out = self.tcn(tcn_input)  # (batch_size, channels, seq_length)
        
        # Take the last time step
        last_hidden = tcn_out[:, :, -1]  # (batch_size, channels)
        
        # Output layers
        out = self.dropout(last_hidden)
        out = self.linear1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.linear2(out)
        
        return out


class CachedScheduledSamplingTCNTrainer:
    """
    Trainer that incorporates cached scheduled sampling for performance and compatibility.
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Cached scheduled sampling parameters
        self.sampling_strategy = config.get('sampling_strategy', 'linear')
        self.noise_factor = config.get('noise_factor', 0.15)
        self.cache_update_frequency = config.get('cache_update_frequency', 5)
        self.max_cache_size = config.get('max_cache_size', 100000)
        
        # Initialize prediction cache
        self.prediction_cache = CachedScheduledSampling(
            cache_update_frequency=self.cache_update_frequency,
            max_cache_size=self.max_cache_size
        )
        
        set_seed(config.get('seed', 42))
    
    def preprocess_data(self, df, target_ride):
        """Preprocess the data for autoregressive training"""
        logger.info(f"Preprocessing data for cached scheduled sampling residual model: {target_ride}")
        
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
    
    def create_features(self, df):
        """Create static feature columns (excluding wait_time and timestamp)"""
        # Static features are everything except wait_time and timestamp
        # wait_time will be used autoregressively, not as a static feature
        static_feature_cols = [col for col in df.columns 
                              if col not in ['wait_time', 'timestamp']]
        
        logger.info(f"Static features: {static_feature_cols}")
        logger.info(f"Total static features: {len(static_feature_cols)}")
        
        return df, static_feature_cols
    
    def load_data_splits(self, splits_output_dir, target_ride):
        """Load train/val/test splits"""
        train_indices = pd.read_parquet(os.path.join(splits_output_dir, "train_indices.parquet"))
        val_indices = pd.read_parquet(os.path.join(splits_output_dir, "validation_indices.parquet"))
        test_indices = pd.read_parquet(os.path.join(splits_output_dir, "test_indices.parquet"))
        
        ride_name_normalized = target_ride.replace(' ', '_')
        train_idx = train_indices[train_indices['ride_name'] == ride_name_normalized]['original_index'].values
        val_idx = val_indices[val_indices['ride_name'] == ride_name_normalized]['original_index'].values
        test_idx = test_indices[test_indices['ride_name'] == ride_name_normalized]['original_index'].values
        
        if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
            raise ValueError(f"No indices found for ride {target_ride}")
        
        logger.info(f"Data splits - Train: {len(train_idx)}, Validation: {len(val_idx)}, Test: {len(test_idx)}")
        return train_idx, val_idx, test_idx
    
    def train_model(self):
        """Train the autoregressive TCN model with cached scheduled sampling"""
        logger.info("Starting cached scheduled sampling autoregressive TCN training...")
        
        # GPU cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Load and preprocess data
        df = pd.read_parquet(self.config['data_path'])
        df = self.preprocess_data(df, self.config['target_ride'])
        
        # Load splits
        train_indices, val_indices, test_indices = self.load_data_splits(
            self.config['splits_output_dir'], 
            self.config['target_ride']
        )
        
        # Create features
        df, static_feature_cols = self.create_features(df)
        
        # Split data
        train_df = df.iloc[train_indices].copy()
        val_df = df.iloc[val_indices].copy()
        test_df = df.iloc[test_indices].copy()
        
        # Prepare features and targets for GradientBoosting
        X_train_static = train_df[static_feature_cols].values
        y_train = train_df['wait_time'].values
        X_val_static = val_df[static_feature_cols].values
        y_val = val_df['wait_time'].values
        X_test_static = test_df[static_feature_cols].values
        y_test = test_df['wait_time'].values
        
        # Train GradientBoosting baseline
        logger.info("Training GradientBoosting baseline...")
        gb_model = GradientBoostingRegressor(
            n_estimators=self.config.get('gb_n_estimators', 100),
            learning_rate=self.config.get('gb_learning_rate', 0.1),
            max_depth=self.config.get('gb_max_depth', 6),
            min_samples_split=self.config.get('gb_min_samples_split', 10),
            min_samples_leaf=self.config.get('gb_min_samples_leaf', 5),
            random_state=42
        )
        
        gb_model.fit(X_train_static, y_train)
        
        # Get baseline predictions and residuals
        y_train_pred_gb = gb_model.predict(X_train_static)
        y_val_pred_gb = gb_model.predict(X_val_static)
        y_test_pred_gb = gb_model.predict(X_test_static)
        
        train_residuals = y_train - y_train_pred_gb
        val_residuals = y_val - y_val_pred_gb
        test_residuals = y_test - y_test_pred_gb
        
        # Evaluate baseline model
        logger.info("Evaluating GradientBoosting baseline...")
        baseline_test_df = test_df.copy()
        baseline_test_df['gb_pred'] = y_test_pred_gb
        
        # Filter out closed rides for evaluation
        if 'closed' in baseline_test_df.columns:
            logger.info(f"Baseline: Excluding {baseline_test_df['closed'].sum()} closed ride data points")
            baseline_open_df = baseline_test_df[baseline_test_df['closed'] == 0]
        else:
            logger.warning("'closed' column not found. Evaluating baseline on all test data.")
            baseline_open_df = baseline_test_df
        
        # Calculate baseline metrics
        y_baseline_actual = baseline_open_df['wait_time'].values
        y_baseline_pred = baseline_open_df['gb_pred'].values
        
        baseline_mae = mean_absolute_error(y_baseline_actual, y_baseline_pred)
        baseline_rmse = np.sqrt(mean_squared_error(y_baseline_actual, y_baseline_pred))
        baseline_r2 = r2_score(y_baseline_actual, y_baseline_pred)
        
        # Calculate baseline sMAPE
        non_zero_mask = y_baseline_actual > 0.1
        if np.sum(non_zero_mask) > 0:
            y_baseline_actual_nz = y_baseline_actual[non_zero_mask]
            y_baseline_pred_nz = y_baseline_pred[non_zero_mask]
            numerator = np.abs(y_baseline_actual_nz - y_baseline_pred_nz)
            denominator = np.abs(y_baseline_actual_nz) + np.abs(y_baseline_pred_nz)
            baseline_smape = np.mean(numerator / denominator) * 100
        else:
            baseline_smape = 0.0
        
        baseline_metrics = {
            "baseline_mae": baseline_mae,
            "baseline_rmse": baseline_rmse,
            "baseline_r2": baseline_r2,
            "baseline_smape": baseline_smape
        }
        
        if wandb.run:
            wandb.log(baseline_metrics)
        
        logger.info(f"Baseline metrics: MAE={baseline_mae:.4f}, RMSE={baseline_rmse:.4f}, R²={baseline_r2:.4f}, sMAPE={baseline_smape:.2f}%")
        
        # Initialize model
        seq_length = self.config['seq_length']
        static_features_size = len(static_feature_cols)
        
        model = AutoregressiveTCNModel(
            static_features_size=static_features_size,
            seq_length=seq_length,
            output_size=1,
            num_channels=self.config['num_channels'],
            kernel_size=self.config['kernel_size'],
            dropout=self.config['dropout'],
            num_layers=self.config.get('num_layers', 8)
        )
        
        model = model.to(self.device)
        
        # Enable torch.compile for performance (now compatible with cached sampling)
        if self.config.get('use_torch_compile', True) and hasattr(torch, 'compile'):
            logger.info("Enabling torch.compile for cached scheduled sampling")
            model = torch.compile(model, mode='reduce-overhead')
        
        # Create datasets
        opening_hour = self.config.get('opening_hour', 9)
        closing_hour = self.config.get('closing_hour', 21)
        total_epochs = self.config['epochs']
        
        train_dataset = CachedScheduledSamplingDataset(
            X_train_static, train_residuals, y_train, seq_length, 
            train_df['timestamp'], opening_hour, closing_hour,
            current_epoch=0, total_epochs=total_epochs,
            sampling_strategy=self.sampling_strategy, noise_factor=self.noise_factor,
            prediction_cache=self.prediction_cache
        )
        
        # Validation dataset doesn't use scheduled sampling (always teacher forcing)
        val_dataset = AutoregressiveResidualsDataset(
            X_val_static, val_residuals, y_val, seq_length,
            val_df['timestamp'], opening_hour, closing_hour
        )
        
        test_dataset = AutoregressiveResidualsDataset(
            X_test_static, test_residuals, y_test, seq_length,
            test_df['timestamp'], opening_hour, closing_hour
        )
        
        # Training setup
        batch_size = self.config['batch_size']
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 1e-5)
        )
        
        # Scheduler
        from torch.optim.lr_scheduler import OneCycleLR
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.config['learning_rate'],
            epochs=self.config['epochs'],
            steps_per_epoch=len(train_dataset) // batch_size + 1,
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        # Mixed precision training (now compatible with cached sampling)
        if self.config.get('use_mixed_precision', True) and torch.cuda.is_available():
            scaler = torch.cuda.amp.GradScaler()
            logger.info("Mixed precision training enabled for cached scheduled sampling")
        else:
            scaler = None
        
        # Training loop with cached scheduled sampling
        best_val_loss = float('inf')
        patience_counter = 0
        patience = self.config.get('patience', 15)
        best_model_state = None
        
        for epoch in range(self.config['epochs']):
            # Update scheduled sampling for this epoch
            train_dataset.update_epoch(epoch)
            
            # Update prediction cache periodically
            if self.prediction_cache.should_update_cache(epoch):
                self.prediction_cache.update_cache(
                    model, gb_model, train_dataset, self.device, static_feature_cols
                )
            
            # Create data loader for this epoch
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                num_workers=4, pin_memory=True if torch.cuda.is_available() else False
            )
            
            # Training phase
            model.train()
            train_loss = 0.0
            train_samples = 0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                
                if scaler is not None:
                    # Mixed precision training
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Standard training
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                
                scheduler.step()
                
                train_loss += loss.item() * inputs.size(0)
                train_samples += inputs.size(0)
            
            train_loss /= train_samples
            
            # Validation phase
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False,
                num_workers=4, pin_memory=True if torch.cuda.is_available() else False
            )
            
            model.eval()
            val_loss = 0.0
            val_samples = 0
            val_residual_preds = []
            val_residual_targets = []
            val_indices_list = []
            
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(val_loader):
                    inputs = inputs.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)
                    
                    if scaler is not None:
                        with torch.cuda.amp.autocast():
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                    
                    val_loss += loss.item() * inputs.size(0)
                    val_samples += inputs.size(0)
                    
                    val_residual_preds.extend(outputs.cpu().numpy().flatten())
                    val_residual_targets.extend(targets.cpu().numpy().flatten())
                    
                    # Track indices for evaluation
                    batch_start = batch_idx * batch_size
                    batch_end = min(batch_start + len(targets), len(val_dataset))
                    val_indices_list.extend(range(batch_start, batch_end))
            
            val_loss /= val_samples
            
            # Calculate combined predictions (GB + TCN residuals) for validation metrics
            val_residual_preds = np.array(val_residual_preds)
            val_residual_targets = np.array(val_residual_targets)
            
            # Get corresponding validation data indices
            val_dataset_indices = [val_dataset.valid_indices[i] for i in val_indices_list]
            val_gb_preds = y_val_pred_gb[val_dataset_indices]
            val_actual_wait_times = y_val[val_dataset_indices]
            
            # Combined predictions = GB baseline + TCN residuals
            val_combined_preds = val_gb_preds + val_residual_preds
            
            # Get evaluation dataframe subset
            val_eval_df = val_df.iloc[val_dataset_indices].reset_index(drop=True)
            
            # Filter out closed rides
            if 'closed' in val_eval_df.columns:
                open_mask = val_eval_df['closed'] == 0
                val_combined_open = val_combined_preds[open_mask]
                val_actual_open = val_actual_wait_times[open_mask]
            else:
                val_combined_open = val_combined_preds
                val_actual_open = val_actual_wait_times
            
            # Calculate validation metrics on combined predictions
            if len(val_combined_open) > 0:
                val_mae = mean_absolute_error(val_actual_open, val_combined_open)
                val_rmse = np.sqrt(mean_squared_error(val_actual_open, val_combined_open))
                val_r2 = r2_score(val_actual_open, val_combined_open)
                
                # Calculate sMAPE
                non_zero_mask = val_actual_open > 0.1
                if np.sum(non_zero_mask) > 0:
                    val_actual_nz = val_actual_open[non_zero_mask]
                    val_combined_nz = val_combined_open[non_zero_mask]
                    numerator = np.abs(val_actual_nz - val_combined_nz)
                    denominator = np.abs(val_actual_nz) + np.abs(val_combined_nz)
                    val_smape = np.mean(numerator / denominator) * 100
                else:
                    val_smape = 0.0
            else:
                val_mae = val_rmse = val_smape = float('inf')
                val_r2 = -float('inf')
            
            # Logging
            current_lr = scheduler.get_last_lr()[0]
            teacher_forcing_prob = train_dataset.teacher_forcing_prob
            cache_size = len(self.prediction_cache.prediction_cache)
            
            metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mae": val_mae,
                "val_rmse": val_rmse,
                "val_r2": val_r2,
                "val_smape": val_smape,
                "learning_rate": current_lr,
                "teacher_forcing_prob": teacher_forcing_prob,
                "cache_size": cache_size,
                "epoch": epoch
            }
            
            if wandb.run:
                wandb.log(metrics)
            
            logger.info(
                f'Epoch {epoch+1}/{self.config["epochs"]} - '
                f'Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, '
                f'Val MAE: {val_mae:.4f}, Val sMAPE: {val_smape:.2f}%, '
                f'TF prob: {teacher_forcing_prob:.3f} ({100-teacher_forcing_prob*100:.1f}% SS active), '
                f'Cache: {cache_size}, LR: {current_lr:.6f}'
            )
            
            # Early stopping based on combined prediction performance
            if teacher_forcing_prob < 0.25:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict().copy()
                    patience_counter = 0
                    logger.info(f"New best model saved at epoch {epoch+1} (TF={teacher_forcing_prob:.3f}, Val Loss={val_loss:.4f})")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch+1} (TF={teacher_forcing_prob:.3f}, 75%+ scheduled sampling achieved)")
                        break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Final evaluation with DUAL approach: both autoregressive and teacher forcing
        logger.info("Performing dual evaluation (autoregressive + teacher forcing)...")
        
        # 1. AUTOREGRESSIVE EVALUATION (Real-world performance)
        logger.info("=== AUTOREGRESSIVE EVALUATION (Real-world performance) ===")
        test_autoregressive_metrics = self._evaluate_autoregressive(
            model, gb_model, test_dataset, batch_size, scaler, test_df, test_indices
        )
        
        # 2. TEACHER FORCING EVALUATION (Ideal conditions performance)
        logger.info("=== TEACHER FORCING EVALUATION (Ideal conditions) ===")
        test_teacher_forcing_metrics = self._evaluate_teacher_forcing(
            model, gb_model, test_dataset, batch_size, scaler, test_df, test_indices
        )
        
        # 3. PERFORMANCE GAP ANALYSIS
        performance_gap = {
            "mae_gap": test_autoregressive_metrics["test_mae"] - test_teacher_forcing_metrics["test_mae_tf"],
            "rmse_gap": test_autoregressive_metrics["test_rmse"] - test_teacher_forcing_metrics["test_rmse_tf"],
            "r2_gap": test_teacher_forcing_metrics["test_r2_tf"] - test_autoregressive_metrics["test_r2"],  # Higher R² is better
            "smape_gap": test_autoregressive_metrics["test_smape"] - test_teacher_forcing_metrics["test_smape_tf"]
        }
        
        # Combine all metrics
        final_metrics = {
            **test_autoregressive_metrics,
            **test_teacher_forcing_metrics,
            **{f"gap_{k}": v for k, v in performance_gap.items()},
            "best_val_loss": best_val_loss
        }
        
        # Log performance comparison
        logger.info("=== DUAL EVALUATION SUMMARY ===")
        logger.info(f"Real-world (Autoregressive) MAE: {test_autoregressive_metrics['test_mae']:.4f}")
        logger.info(f"Ideal conditions (Teacher Forcing) MAE: {test_teacher_forcing_metrics['test_mae_tf']:.4f}")
        logger.info(f"Performance gap (Real - Ideal): {performance_gap['mae_gap']:.4f}")
        logger.info(f"Gap percentage: {(performance_gap['mae_gap']/test_teacher_forcing_metrics['test_mae_tf']*100):.1f}%")
        
        if wandb.run:
            wandb.log(final_metrics)
        
        logger.info(f"Final dual evaluation metrics: {final_metrics}")
        
        # Save models
        self._save_models(gb_model, model)
        
        # Cleanup
        self.prediction_cache.clear_cache()
        del model, gb_model, optimizer, scheduler, criterion
        del train_dataset, val_dataset, test_dataset
        del train_loader, val_loader
        
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return final_metrics
    
    def _evaluate_autoregressive(self, model, gb_model, test_dataset, batch_size, scaler, test_df, test_indices):
        """
        Evaluate model performance using autoregressive prediction (real-world conditions).
        Fixed version that properly builds sequences matching training distribution.
        """
        logger.info("Evaluating with autoregressive prediction (model uses own predictions)...")
        
        model.eval()
        autoregressive_preds = []
        actual_wait_times = []
        test_indices_used = []
        
        # Create a copy of test dataset with high scheduled sampling for autoregressive evaluation
        autoregressive_dataset = CachedScheduledSamplingDataset(
            test_dataset.X_static, 
            test_dataset.residuals, 
            test_dataset.wait_times, 
            test_dataset.seq_length,
            test_dataset.timestamps, 
            test_dataset.opening_hour, 
            test_dataset.closing_hour,
            current_epoch=99,  # High epoch = mostly scheduled sampling (5% teacher forcing)
            total_epochs=100,
            sampling_strategy="linear",
            noise_factor=0.15,
            prediction_cache=self.prediction_cache
        )
        
        # Process samples in smaller batches to avoid memory issues with autoregressive generation
        autoregressive_batch_size = min(32, batch_size)  # Smaller batches for autoregressive
        
        with torch.no_grad():
            for start_idx in range(0, len(autoregressive_dataset), autoregressive_batch_size):
                end_idx = min(start_idx + autoregressive_batch_size, len(autoregressive_dataset))
                
                batch_preds = []
                batch_actuals = []
                batch_indices = []
                
                for idx in range(start_idx, end_idx):
                    try:
                        # Get autoregressive sequence (uses cached predictions + some teacher forcing)
                        inputs, targets = autoregressive_dataset[idx]
                        
                        # Model prediction
                        inputs = inputs.unsqueeze(0).to(self.device)
                        if scaler is not None:
                            with torch.cuda.amp.autocast():
                                residual_pred = model(inputs).cpu().numpy().flatten()[0]
                        else:
                            residual_pred = model(inputs).cpu().numpy().flatten()[0]
                        
                        # Get corresponding dataset index and static features
                        dataset_idx = autoregressive_dataset.valid_indices[idx]
                        static_features = autoregressive_dataset.X_static[dataset_idx]
                        
                        # Get baseline prediction
                        gb_pred = gb_model.predict(static_features.reshape(1, -1))[0]
                        
                        # Combined prediction
                        combined_pred = gb_pred + residual_pred
                        actual_wait_time = autoregressive_dataset.wait_times[dataset_idx]
                        
                        batch_preds.append(combined_pred)
                        batch_actuals.append(actual_wait_time)
                        batch_indices.append(dataset_idx)
                        
                    except Exception as e:
                        logger.debug(f"Autoregressive evaluation failed for sample {idx}: {e}")
                        continue
                
                # Accumulate results
                autoregressive_preds.extend(batch_preds)
                actual_wait_times.extend(batch_actuals)
                test_indices_used.extend(batch_indices)
                
                # Log progress every 1000 samples
                if len(autoregressive_preds) % 1000 == 0:
                    logger.info(f"Processed {len(autoregressive_preds)} autoregressive samples...")
        
        # Convert to arrays
        autoregressive_preds = np.array(autoregressive_preds)
        actual_wait_times = np.array(actual_wait_times)
        
        logger.info(f"Generated {len(autoregressive_preds)} autoregressive predictions")
        
        # Create evaluation dataframe
        eval_df = test_df.iloc[test_indices_used].reset_index(drop=True)
        eval_df['autoregressive_pred'] = autoregressive_preds
        eval_df['actual_wait_time'] = actual_wait_times
        
        # Filter out closed rides
        if 'closed' in eval_df.columns:
            logger.info(f"Autoregressive: Excluding {eval_df['closed'].sum()} closed ride data points")
            open_df = eval_df[eval_df['closed'] == 0]
        else:
            logger.warning("'closed' column not found. Evaluating on all test data.")
            open_df = eval_df
        
        # Calculate metrics
        if len(open_df) == 0:
            logger.error("No open ride data points for autoregressive evaluation!")
            return {
                "test_mae": float('inf'),
                "test_rmse": float('inf'),
                "test_r2": -float('inf'),
                "test_smape": float('inf'),
                "autoregressive_samples": 0
            }
        
        y_actual = open_df['actual_wait_time'].values
        y_pred = open_df['autoregressive_pred'].values
        
        # Remove any infinite or NaN predictions
        valid_mask = np.isfinite(y_pred) & np.isfinite(y_actual)
        if not np.all(valid_mask):
            logger.warning(f"Removing {np.sum(~valid_mask)} invalid predictions")
            y_actual = y_actual[valid_mask]
            y_pred = y_pred[valid_mask]
        
        if len(y_actual) == 0:
            logger.error("No valid predictions for autoregressive evaluation!")
            return {
                "test_mae": float('inf'),
                "test_rmse": float('inf'),
                "test_r2": -float('inf'),
                "test_smape": float('inf'),
                "autoregressive_samples": 0
            }
        
        mae = mean_absolute_error(y_actual, y_pred)
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        r2 = r2_score(y_actual, y_pred)
        
        # Calculate sMAPE
        non_zero_mask = y_actual > 0.1
        if np.sum(non_zero_mask) > 0:
            y_actual_nz = y_actual[non_zero_mask]
            y_pred_nz = y_pred[non_zero_mask]
            numerator = np.abs(y_actual_nz - y_pred_nz)
            denominator = np.abs(y_actual_nz) + np.abs(y_pred_nz)
            smape = np.mean(numerator / denominator) * 100
        else:
            smape = 0.0
        
        autoregressive_metrics = {
            "test_mae": mae,
            "test_rmse": rmse,
            "test_r2": r2,
            "test_smape": smape,
            "autoregressive_samples": len(y_actual)
        }
        
        logger.info(f"Autoregressive metrics: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}, sMAPE={smape:.2f}%")
        
        # Log prediction statistics for debugging
        logger.info(f"Prediction stats: mean={np.mean(y_pred):.2f}, std={np.std(y_pred):.2f}, min={np.min(y_pred):.2f}, max={np.max(y_pred):.2f}")
        logger.info(f"Actual stats: mean={np.mean(y_actual):.2f}, std={np.std(y_actual):.2f}, min={np.min(y_actual):.2f}, max={np.max(y_actual):.2f}")
        
        return autoregressive_metrics
    
    def _evaluate_teacher_forcing(self, model, gb_model, test_dataset, batch_size, scaler, test_df, test_indices):
        """
        Evaluate model performance using teacher forcing (ideal conditions).
        Model uses ground truth inputs, showing theoretical best performance.
        """
        logger.info("Evaluating with teacher forcing (model uses ground truth)...")
        
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True if torch.cuda.is_available() else False
        )
        
        model.eval()
        test_residual_preds = []
        test_indices_list = []
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(self.device, non_blocking=True)
                
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)
                
                test_residual_preds.extend(outputs.cpu().numpy().flatten())
                
                # Track indices
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + len(targets), len(test_dataset))
                test_indices_list.extend(range(batch_start, batch_end))
        
        # Calculate combined predictions
        test_residual_preds = np.array(test_residual_preds)
        test_dataset_indices = [test_dataset.valid_indices[i] for i in test_indices_list]
        
        # Get GB baseline predictions for these indices
        test_static_features = test_df.iloc[test_dataset_indices][
            [col for col in test_df.columns if col not in ['wait_time', 'timestamp']]
        ].values
        test_gb_preds = gb_model.predict(test_static_features)
        test_actual_wait_times = test_df.iloc[test_dataset_indices]['wait_time'].values
        
        # Combined predictions = GB baseline + TCN residuals
        test_combined_preds = test_gb_preds + test_residual_preds
        
        # Evaluation dataframe
        eval_df = test_df.iloc[test_dataset_indices].reset_index(drop=True)
        eval_df['teacher_forcing_pred'] = test_combined_preds
        eval_df['actual_wait_time'] = test_actual_wait_times
        
        # Filter out closed rides
        if 'closed' in eval_df.columns:
            logger.info(f"Teacher forcing: Excluding {eval_df['closed'].sum()} closed ride data points")
            open_df = eval_df[eval_df['closed'] == 0]
        else:
            logger.warning("'closed' column not found. Evaluating on all test data.")
            open_df = eval_df
        
        # Calculate metrics
        y_actual = open_df['actual_wait_time'].values
        y_pred = open_df['teacher_forcing_pred'].values
        
        mae = mean_absolute_error(y_actual, y_pred)
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        r2 = r2_score(y_actual, y_pred)
        
        # Calculate sMAPE
        non_zero_mask = y_actual > 0.1
        if np.sum(non_zero_mask) > 0:
            y_actual_nz = y_actual[non_zero_mask]
            y_pred_nz = y_pred[non_zero_mask]
            numerator = np.abs(y_actual_nz - y_pred_nz)
            denominator = np.abs(y_actual_nz) + np.abs(y_pred_nz)
            smape = np.mean(numerator / denominator) * 100
        else:
            smape = 0.0
        
        teacher_forcing_metrics = {
            "test_mae_tf": mae,
            "test_rmse_tf": rmse,
            "test_r2_tf": r2,
            "test_smape_tf": smape,
            "teacher_forcing_samples": len(y_actual)
        }
        
        logger.info(f"Teacher forcing metrics: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}, sMAPE={smape:.2f}%")
        return teacher_forcing_metrics
    
    def _save_models(self, gb_model, tcn_model):
        """Save both GradientBoosting and TCN models"""
        os.makedirs("models/cached_scheduled_sampling", exist_ok=True)
        target_ride = self.config['target_ride'].replace(' ', '_')
        
        # Save GradientBoosting model
        gb_path = f"models/cached_scheduled_sampling/{target_ride}_gb_baseline.pkl"
        with open(gb_path, "wb") as f:
            pickle.dump(gb_model, f)
        
        # Save TCN model with config
        tcn_path = f"models/cached_scheduled_sampling/{target_ride}_cached_scheduled_sampling_tcn.pt"
        torch.save({
            'model_state_dict': tcn_model.state_dict(),
            'config': self.config,
            'cache_config': {
                'cache_update_frequency': self.cache_update_frequency,
                'max_cache_size': self.max_cache_size,
                'sampling_strategy': self.sampling_strategy,
                'noise_factor': self.noise_factor
            }
        }, tcn_path)
        
        logger.info(f"Models saved: {gb_path}, {tcn_path}")
        
        # Log to wandb
        if wandb.run:
            gb_artifact = wandb.Artifact(f"gb_baseline_cached_ss_{wandb.run.id}", type="model")
            gb_artifact.add_file(gb_path)
            wandb.log_artifact(gb_artifact)
            
            tcn_artifact = wandb.Artifact(f"cached_scheduled_sampling_tcn_{wandb.run.id}", type="model")
            tcn_artifact.add_file(tcn_path)
            wandb.log_artifact(tcn_artifact)


def load_config(config_path):
    """Load configuration from YAML or JSON file"""
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            return yaml.safe_load(f)
        else:
            return json.load(f)


def create_config_from_ride(ride_name, rides_config_path="configs/rides_config.yaml"):
    """Create a training config from ride name and rides config"""
    with open(rides_config_path, 'r') as f:
        rides_config = yaml.safe_load(f)
    
    if ride_name not in rides_config['rides']:
        raise ValueError(f"Ride '{ride_name}' not found in {rides_config_path}")
    
    # Start with default parameters
    config = rides_config['default_params'].copy()
    
    # Add global settings
    config.update(rides_config['global_settings'])
    
    # Add ride-specific settings
    ride_info = rides_config['rides'][ride_name]
    config['data_path'] = ride_info['data_path']
    config['target_ride'] = ride_name
    
    # Autoregressive-specific defaults with cached scheduled sampling
    autoregressive_defaults = {
        'seq_length': 96,        # 24 hours with 15-min intervals
        'batch_size': 128,
        'num_channels': 256,
        'kernel_size': 3,
        'num_layers': 8,
        'dropout': 0.3,
        'learning_rate': 0.0005,
        'weight_decay': 1e-5,
        'opening_hour': 9,
        'closing_hour': 21,
        # GradientBoosting parameters
        'gb_n_estimators': 100,
        'gb_learning_rate': 0.1,
        'gb_max_depth': 6,
        'gb_min_samples_split': 10,
        'gb_min_samples_leaf': 5,
        # Cached scheduled sampling parameters
        'sampling_strategy': 'linear',  # 'linear', 'exponential', 'inverse_sigmoid'
        'noise_factor': 0.15,           # Standard deviation factor for prediction noise
        'cache_update_frequency': 5,     # Update cache every N epochs
        'max_cache_size': 100000,       # Maximum cached predictions
        'use_torch_compile': True,      # Enable torch.compile
        'use_mixed_precision': True,    # Enable mixed precision
        'run_name': 'cached_scheduled_sampling_autoregressive'
    }
    
    for key, value in autoregressive_defaults.items():
        if key not in config:
            config[key] = value
    
    return config


def main():
    parser = argparse.ArgumentParser(description="Train autoregressive TCN with cached scheduled sampling")
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--ride', help='Ride name (alternative to --config)')
    parser.add_argument('--rides-config', default='configs/rides_config.yaml',
                       help='Path to rides configuration file')
    parser.add_argument('--wandb-project', default='waitless-cached-scheduled-sampling-tcn-hslu-dspro2-fs25',
                       help='Weights & Biases project name')
    parser.add_argument('--wandb-entity', default='waitless-hslu-dspro2-fs25',
                       help='Weights & Biases entity name')
    parser.add_argument('--sampling-strategy', choices=['linear', 'exponential', 'inverse_sigmoid'],
                       help='Scheduled sampling strategy')
    parser.add_argument('--noise-factor', type=float, 
                       help='Noise factor for prediction uncertainty')
    parser.add_argument('--cache-update-frequency', type=int,
                       help='Update prediction cache every N epochs')
    parser.add_argument('--max-cache-size', type=int,
                       help='Maximum number of cached predictions')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    elif args.ride:
        config = create_config_from_ride(args.ride, args.rides_config)
        logger.info(f"Created cached scheduled sampling configuration for ride: {args.ride}")
    else:
        raise ValueError("Either --config or --ride must be specified")
    
    # Override config with command line arguments if provided
    if args.sampling_strategy:
        config['sampling_strategy'] = args.sampling_strategy
    if args.noise_factor:
        config['noise_factor'] = args.noise_factor
    if args.cache_update_frequency:
        config['cache_update_frequency'] = args.cache_update_frequency
    if args.max_cache_size:
        config['max_cache_size'] = args.max_cache_size
    
    # Initialize wandb
    if config.get('use_wandb', True):
        run_name = f"{config['target_ride']}_cached_scheduled_sampling_tcn"
        if config.get('run_name'):
            run_name = f"{config['target_ride']}_{config['run_name']}"
        
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=config,
            name=run_name,
            tags=['cached_scheduled_sampling', 'autoregressive', 'tcn', 'gradientboosting', config['target_ride']]
        )
    
    # Train model
    trainer = CachedScheduledSamplingTCNTrainer(config)
    metrics = trainer.train_model()
    
    if wandb.run:
        wandb.finish()
    
    logger.info("Cached scheduled sampling autoregressive TCN training completed!")


if __name__ == "__main__":
    main()