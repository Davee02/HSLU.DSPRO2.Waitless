import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class AutoregressiveResidualsDataset(Dataset):
    """
    Original dataset for autoregressive residual prediction (used for validation/test).
    """
    def __init__(self, X_static, residuals, wait_times, seq_length, timestamps, 
                 opening_hour=11, closing_hour=17):
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