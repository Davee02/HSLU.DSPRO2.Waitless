import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .data_utils import calculate_scheduled_sampling_probability

logger = logging.getLogger(__name__)


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
        return calculate_scheduled_sampling_probability(
            self.current_epoch, self.total_epochs, self.sampling_strategy
        )
    
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