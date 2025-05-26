#!/usr/bin/env python3
"""
Autoregressive TCN Training Script for Wait Time Prediction
Hybrid approach: GradientBoosting baseline + Autoregressive TCN for residuals
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
        logging.FileHandler('autoregressive_training.log'),
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

class AutoregressiveResidualsDataset(Dataset):
    """
    Dataset for autoregressive residual prediction.
    Uses previous residuals/wait_times as features while avoiding data leakage.
    """
    def __init__(self, X_static, residuals, wait_times, seq_length, timestamps, 
                 opening_hour=9, closing_hour=21):
        self.X_static = X_static  # Static features (non-temporal)
        self.residuals = residuals  # Target residuals to predict
        self.wait_times = wait_times  # Actual wait times (for autoregressive features)
        self.seq_length = seq_length
        
        if isinstance(timestamps, pd.Series):
            self.timestamps = pd.to_datetime(timestamps.reset_index(drop=True))
        else:
            self.timestamps = pd.to_datetime(timestamps)
        
        self.opening_hour = opening_hour
        self.closing_hour = closing_hour
        
        # for same-day logic
        self.dates = self.timestamps.dt.date
        
        self.valid_indices = self._create_valid_indices()
        
        logger.info(f"Autoregressive residuals dataset: {len(self.valid_indices)} valid sequences")
    
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
        Create sequence with autoregressive logic for residuals prediction.
        Uses previous wait_times and residuals as features, handling same-day predictions.
        """
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
            
            # Same-day prediction logic: use simulated values after opening time
            if (seq_date == target_date and 
                seq_timestamp.hour >= self.opening_hour):
                
                # For same-day predictions, simulate prediction uncertainty
                # Add noise to both wait_time and residual
                wait_noise_std = max(0.1, actual_wait_time * 0.15)
                residual_noise_std = max(0.05, abs(actual_residual) * 0.2)
                
                simulated_wait_time = actual_wait_time + np.random.normal(0, wait_noise_std)
                simulated_residual = actual_residual + np.random.normal(0, residual_noise_std)

                simulated_wait_time = max(0, simulated_wait_time)
                
                autoregressive_features.extend([simulated_wait_time, simulated_residual])
            else:
                # Use actual historical values
                autoregressive_features.extend([actual_wait_time, actual_residual])
        
        # Combine static and autoregressive features
        # Static features + sequence of [wait_time, residual] pairs
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

class AutoregressiveTCNTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        set_seed(config.get('seed', 42))
    
    def preprocess_data(self, df, target_ride):
        """Preprocess the data for autoregressive training"""
        logger.info(f"Preprocessing data for autoregressive residual model: {target_ride}")
        
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
        """Train the autoregressive TCN model with GradientBoosting baseline"""
        logger.info("Starting autoregressive TCN training with GradientBoosting baseline...")
        
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
        
        # Get baseline predictions
        y_train_pred_gb = gb_model.predict(X_train_static)
        y_val_pred_gb = gb_model.predict(X_val_static)
        y_test_pred_gb = gb_model.predict(X_test_static)
        
        # Calculate residuals
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
        
        # Create autoregressive datasets for residuals
        seq_length = self.config['seq_length']
        opening_hour = self.config.get('opening_hour', 9)
        closing_hour = self.config.get('closing_hour', 21)
        
        train_dataset = AutoregressiveResidualsDataset(
            X_train_static, train_residuals, y_train, seq_length, 
            train_df['timestamp'], opening_hour, closing_hour
        )
        val_dataset = AutoregressiveResidualsDataset(
            X_val_static, val_residuals, y_val, seq_length,
            val_df['timestamp'], opening_hour, closing_hour
        )
        test_dataset = AutoregressiveResidualsDataset(
            X_test_static, test_residuals, y_test, seq_length,
            test_df['timestamp'], opening_hour, closing_hour
        )
        
        # Create data loaders
        batch_size = self.config['batch_size']
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=2, pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=2, pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=2, pin_memory=True if torch.cuda.is_available() else False
        )
        
        # Initialize autoregressive TCN model
        logger.info("Initializing autoregressive TCN for residuals...")
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
        
        # Model compilation
        if hasattr(torch, 'compile') and torch.cuda.is_available():
            try:
                model = torch.compile(model, mode='default')
                logger.info("Model compiled with torch.compile")
            except Exception as e:
                logger.info(f"torch.compile not available: {e}")
        
        # Training setup
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
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        # Mixed precision training
        scaler = torch.amp.GradScaler() if torch.cuda.is_available() else None
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        patience = self.config.get('patience', 15)
        best_model_state = None
        
        for epoch in range(self.config['epochs']):
            # Training phase
            model.train()
            train_loss = 0.0
            train_samples = 0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                
                scheduler.step()
                
                train_loss += loss.item() * inputs.size(0)
                train_samples += inputs.size(0)
            
            train_loss /= train_samples
            
            # Validation phase
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
                        with torch.amp.autocast('cuda'):
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
            metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mae": val_mae,
                "val_rmse": val_rmse,
                "val_r2": val_r2,
                "val_smape": val_smape,
                "learning_rate": current_lr
            }
            
            if wandb.run:
                wandb.log(metrics)
            
            logger.info(
                f'Epoch {epoch+1}/{self.config["epochs"]} - '
                f'Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, '
                f'Val MAE: {val_mae:.4f}, Val sMAPE: {val_smape:.2f}%, '
                f'LR: {current_lr:.6f}'
            )
            
            # Early stopping based on combined prediction performance
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Final evaluation
        logger.info("Performing final evaluation...")
        model.eval()
        test_residual_preds = []
        test_indices_list = []
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(self.device, non_blocking=True)
                
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)
                
                test_residual_preds.extend(outputs.cpu().numpy().flatten())
                
                # Track indices
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + len(targets), len(test_dataset))
                test_indices_list.extend(range(batch_start, batch_end))
        
        # Calculate final combined predictions
        test_residual_preds = np.array(test_residual_preds)
        test_dataset_indices = [test_dataset.valid_indices[i] for i in test_indices_list]
        test_gb_preds = y_test_pred_gb[test_dataset_indices]
        test_actual_wait_times = y_test[test_dataset_indices]
        
        # Combined predictions = GB baseline + TCN residuals
        test_combined_preds = test_gb_preds + test_residual_preds
        
        # Evaluation dataframe
        test_eval_df = test_df.iloc[test_dataset_indices].reset_index(drop=True)
        test_eval_df['gb_pred'] = test_gb_preds
        test_eval_df['residual_pred'] = test_residual_preds
        test_eval_df['combined_pred'] = test_combined_preds
        
        # Filter out closed rides
        if 'closed' in test_eval_df.columns:
            logger.info(f"Final evaluation: Excluding {test_eval_df['closed'].sum()} closed ride data points")
            test_open_df = test_eval_df[test_eval_df['closed'] == 0]
        else:
            logger.warning("'closed' column not found. Evaluating on all test data.")
            test_open_df = test_eval_df
        
        # Calculate final metrics
        y_test_actual = test_open_df['wait_time'].values
        y_test_combined = test_open_df['combined_pred'].values
        
        test_mae = mean_absolute_error(y_test_actual, y_test_combined)
        test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_combined))
        test_r2 = r2_score(y_test_actual, y_test_combined)
        
        # Calculate test sMAPE
        non_zero_mask = y_test_actual > 0.1
        if np.sum(non_zero_mask) > 0:
            y_test_actual_nz = y_test_actual[non_zero_mask]
            y_test_combined_nz = y_test_combined[non_zero_mask]
            numerator = np.abs(y_test_actual_nz - y_test_combined_nz)
            denominator = np.abs(y_test_actual_nz) + np.abs(y_test_combined_nz)
            test_smape = np.mean(numerator / denominator) * 100
        else:
            test_smape = 0.0
        
        final_metrics = {
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_r2": test_r2,
            "test_smape": test_smape,
            "best_val_loss": best_val_loss
        }
        
        if wandb.run:
            wandb.log(final_metrics)
        
        logger.info(f"Final test metrics: {final_metrics}")
        
        # Save models
        self._save_models(gb_model, model)
        
        # Cleanup
        del model, gb_model, optimizer, scheduler, criterion
        del train_dataset, val_dataset, test_dataset
        del train_loader, val_loader, test_loader
        
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return final_metrics
    
    def _save_models(self, gb_model, tcn_model):
        """Save both GradientBoosting and TCN models"""
        os.makedirs("models/autoregressive", exist_ok=True)
        target_ride = self.config['target_ride'].replace(' ', '_')
        
        # Save GradientBoosting model
        gb_path = f"models/autoregressive/{target_ride}_gb_baseline.pkl"
        with open(gb_path, "wb") as f:
            pickle.dump(gb_model, f)
        
        # Save TCN model with config
        tcn_path = f"models/autoregressive/{target_ride}_autoregressive_tcn.pt"
        torch.save({
            'model_state_dict': tcn_model.state_dict(),
            'config': self.config
        }, tcn_path)
        
        logger.info(f"Models saved: {gb_path}, {tcn_path}")
        
        # Log to wandb
        if wandb.run:
            gb_artifact = wandb.Artifact(f"gb_baseline_{wandb.run.id}", type="model")
            gb_artifact.add_file(gb_path)
            wandb.log_artifact(gb_artifact)
            
            tcn_artifact = wandb.Artifact(f"autoregressive_tcn_{wandb.run.id}", type="model")
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
    
    # Autoregressive-specific defaults
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
        'run_name': 'autoregressive'
    }
    
    for key, value in autoregressive_defaults.items():
        if key not in config:
            config[key] = value
    
    return config

def main():
    parser = argparse.ArgumentParser(description="Train autoregressive TCN with GradientBoosting baseline")
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--ride', help='Ride name (alternative to --config)')
    parser.add_argument('--rides-config', default='configs/rides_config.yaml',
                       help='Path to rides configuration file')
    parser.add_argument('--wandb-project', default='waitless-autoregressive-tcn-hslu-dspro2-fs25',
                       help='Weights & Biases project name')
    parser.add_argument('--wandb-entity', default='waitless-hslu-dspro2-fs25',
                       help='Weights & Biases entity name')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    elif args.ride:
        config = create_config_from_ride(args.ride, args.rides_config)
        logger.info(f"Created autoregressive configuration for ride: {args.ride}")
    else:
        raise ValueError("Either --config or --ride must be specified")
    
    # Initialize wandb
    if config.get('use_wandb', True):
        run_name = f"{config['target_ride']}_autoregressive_gb_tcn"
        if config.get('run_name'):
            run_name = f"{config['target_ride']}_{config['run_name']}"
        
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=config,
            name=run_name,
            tags=['autoregressive', 'tcn', 'gradientboosting', config['target_ride']]
        )
    
    # Train model
    trainer = AutoregressiveTCNTrainer(config)
    metrics = trainer.train_model()
    
    if wandb.run:
        wandb.finish()
    
    logger.info("Autoregressive TCN training with GradientBoosting baseline completed!")

if __name__ == "__main__":
    main()