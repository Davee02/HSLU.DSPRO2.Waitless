#!/usr/bin/env python3
"""
TCN Training Script for Wait Time Prediction
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
import yaml

import wandb
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import pickle
from pytorch_tcn import TCN
from sklearn.ensemble import GradientBoostingRegressor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Reproducibility
def set_seed(seed=42):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, seq_length):
        self.X = X
        self.y = y
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.X) - self.seq_length
    
    def __getitem__(self, idx):
        X_seq = self.X[idx:idx + self.seq_length]
        y_value = self.y[idx + self.seq_length]
        return torch.FloatTensor(X_seq), torch.FloatTensor([y_value])

class TCNModel(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout=0.2):
        super(TCNModel, self).__init__()
        self.tcn = TCN(
            num_inputs=input_size,         
            num_channels=[num_channels] * 8,
            kernel_size=kernel_size,        
            dropout=dropout,                
            causal=True,                    
            use_skip_connections=True       
        )
        self.linear = nn.Linear(num_channels, output_size)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        y = self.tcn(x)
        y = y[:, :, -1]
        return self.linear(y)

class TCNTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        set_seed(config.get('seed', 42))
        
    def preprocess_data(self, df, target_ride):
        """Preprocess the data for a single ride"""
        if 'time_bucket' in df.columns:
            df = df.drop(columns=['time_bucket'])
        
        logger.info(f"Building model for ride: {target_ride}")
        
        ride_col = f'ride_name_{target_ride}'
        if ride_col in df.columns:
            df = df[df[ride_col] == 1].copy()
        
        ride_cols = [col for col in df.columns if col.startswith('ride_name_')]
        df = df.drop(columns=ride_cols)
        df = df.fillna(0)
        
        return df

    def create_features(self, df):
        """Create features for the model"""
        feature_cols = [col for col in df.columns if col not in ['wait_time', 'timestamp']]
        return df, feature_cols

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
            raise ValueError(f"No indices found for ride {target_ride}. Check ride name or indices files.")
        
        logger.info(f"Found {len(train_idx)} train, {len(val_idx)} validation, and {len(test_idx)} test samples")
        return train_idx, val_idx, test_idx

    def train_single_model(self):
        """Train a single model with given configuration"""
        # Aggressive GPU cleanup at start
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Load data
        df = pd.read_parquet(self.config['data_path'])
        df = self.preprocess_data(df, self.config['target_ride'])
        
        # Load splits
        train_indices, val_indices, test_indices = self.load_data_splits(
            self.config['splits_output_dir'], 
            self.config['target_ride']
        )
        
        # Create features
        df, feature_cols = self.create_features(df)
        
        # Split data
        train_df = df.iloc[train_indices].copy()
        val_df = df.iloc[val_indices].copy()
        test_df = df.iloc[test_indices].copy()
        
        # Prepare features and target
        X_train = train_df[feature_cols].values
        y_train = train_df['wait_time'].values
        X_val = val_df[feature_cols].values
        y_val = val_df['wait_time'].values
        X_test = test_df[feature_cols].values
        y_test = test_df['wait_time'].values
        
        # Train linear model
        logger.info("Training linear model...")
        linear_model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
        )
        
        linear_model.fit(X_train, y_train)
        
        # Get predictions and residuals
        y_train_pred_linear = linear_model.predict(X_train)
        y_val_pred_linear = linear_model.predict(X_val)
        y_test_pred_linear = linear_model.predict(X_test)


        # Evaluate baseline model performance
        logger.info("Evaluating baseline model...")
        
        # Create test evaluation dataframe for baseline
        baseline_test_df = test_df.copy()
        baseline_test_df['baseline_pred'] = y_test_pred_linear
        
        # Filter out closed rides if column exists
        if 'closed' in baseline_test_df.columns:
            logger.info(f"Baseline evaluation: Excluding {baseline_test_df['closed'].sum()} data points where ride is closed")
            baseline_open_df = baseline_test_df[baseline_test_df['closed'] == 0]
        else:
            logger.warning("'closed' column not found. Evaluating baseline on all test data.")
            baseline_open_df = baseline_test_df
        
        # Calculate baseline metrics
        y_baseline_actual = baseline_open_df['wait_time'].values
        y_baseline_pred = baseline_open_df['baseline_pred'].values
        
        baseline_mae = mean_absolute_error(y_baseline_actual, y_baseline_pred)
        baseline_rmse = np.sqrt(mean_squared_error(y_baseline_actual, y_baseline_pred))
        baseline_r2 = r2_score(y_baseline_actual, y_baseline_pred)
        
        # Calculate baseline sMAPE
        non_zero_mask = y_baseline_actual > 0.1
        if np.sum(non_zero_mask) > 0:
            y_baseline_actual_nonzero = y_baseline_actual[non_zero_mask]
            y_baseline_pred_nonzero = y_baseline_pred[non_zero_mask]
            
            numerator = np.abs(y_baseline_actual_nonzero - y_baseline_pred_nonzero)
            denominator = np.abs(y_baseline_actual_nonzero) + np.abs(y_baseline_pred_nonzero)
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
        
        logger.info(f"Baseline model metrics: MAE={baseline_mae:.4f}, RMSE={baseline_rmse:.4f}, R²={baseline_r2:.4f}, sMAPE={baseline_smape:.2f}%")
               
        train_residuals = y_train - y_train_pred_linear
        val_residuals = y_val - y_val_pred_linear
        test_residuals = y_test - y_test_pred_linear
        
        # Create datasets
        seq_length = self.config['seq_length']
        train_dataset = TimeSeriesDataset(X_train, train_residuals, seq_length)
        val_dataset = TimeSeriesDataset(X_val, val_residuals, seq_length)
        test_dataset = TimeSeriesDataset(X_test, test_residuals, seq_length)
        
        # Create data loaders with performance optimizations
        batch_size = self.config['batch_size']
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=2,  # Parallel data loading
            pin_memory=True if torch.cuda.is_available() else False,  # Faster GPU transfer
            persistent_workers=True  # Keep workers alive between epochs
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        # Initialize TCN model
        logger.info("Initializing TCN model...")
        input_size = X_train.shape[1]
        tcn_model = TCNModel(
            input_size=input_size,
            output_size=1,
            num_channels=self.config['num_channels'],
            kernel_size=self.config['kernel_size'],
            dropout=self.config['dropout']
        )
        
        # Move model to device AFTER creation
        tcn_model = tcn_model.to(self.device)
        
        # Performance optimization: compile model if using PyTorch 2.0+
        if hasattr(torch, 'compile') and torch.cuda.is_available():
            try:
                tcn_model = torch.compile(tcn_model, mode='default')
                logger.info("Model compiled with torch.compile for better performance")
            except Exception as e:
                logger.info(f"torch.compile not available or failed: {e}")
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(tcn_model.parameters(), lr=self.config['learning_rate'])
        
        # Use AMP for faster training on modern GPUs
        scaler = torch.amp.GradScaler() if torch.cuda.is_available() else None
        
        # Scheduler
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.config.get('t_max', self.config['epochs']),
            eta_min=self.config.get('eta_min', 1e-6)
        )
        
        # Training loop
        logger.info("Starting TCN training...")
        best_val_loss = float('inf')
        counter = 0
        best_model = None
        patience = self.config.get('patience', 10)
        
        for epoch in range(self.config['epochs']):
            # Training phase
            tcn_model.train()
            train_loss = 0.0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                
                # Use mixed precision if available
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        outputs = tcn_model(inputs)
                        loss = criterion(outputs, targets)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = tcn_model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    
                train_loss += loss.item()
                
            train_loss /= len(train_loader)
            
            # Validation phase
            tcn_model.eval()
            val_loss = 0.0
            val_predictions = []
            val_targets = []
            val_indices = []  # Track which samples we're using
            
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(val_loader):
                    inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
                    
                    # Use mixed precision for validation too
                    if scaler is not None:
                        with torch.amp.autocast('cuda'):
                            outputs = tcn_model(inputs)
                            loss = criterion(outputs, targets)
                    else:
                        outputs = tcn_model(inputs)
                        loss = criterion(outputs, targets)
                        
                    val_loss += loss.item()
                    
                    # Collect predictions, targets, and indices for filtering
                    val_predictions.extend(outputs.cpu().numpy().flatten())
                    val_targets.extend(targets.cpu().numpy().flatten())
                    # Calculate the actual indices in the validation set
                    batch_start = batch_idx * batch_size
                    batch_end = min(batch_start + len(targets), len(val_dataset))
                    val_indices.extend(range(batch_start, batch_end))
                    
            val_loss /= len(val_loader)
            
            # Filter out closed ride times for metric calculation (same as final evaluation)
            val_predictions = np.array(val_predictions)
            val_targets = np.array(val_targets)
            val_indices = np.array(val_indices)
            
            # Get the corresponding validation data (accounting for sequence length)
            val_eval_indices = val_indices + seq_length  # Adjust for sequence offset
            val_eval_df = val_df.iloc[val_eval_indices].reset_index(drop=True)
            
            # Filter out closed rides if 'closed' column exists
            if 'closed' in val_eval_df.columns:
                open_mask = val_eval_df['closed'] == 0
                val_predictions_open = val_predictions[open_mask]
                val_targets_open = val_targets[open_mask]
            else:
                val_predictions_open = val_predictions
                val_targets_open = val_targets
            
            # Calculate metrics only on open ride times
            if len(val_predictions_open) > 0:
                # MAE and RMSE are calculated on residuals (correct)
                val_mae = mean_absolute_error(val_targets_open, val_predictions_open)
                val_rmse = np.sqrt(mean_squared_error(val_targets_open, val_predictions_open))
                
                # For sMAPE, we need actual wait times vs combined predictions
                # The validation predictions are residuals, so we need to add them to linear predictions
                
                # Get the subset of linear predictions that correspond to our validation samples
                val_linear_all = y_val_pred_linear[seq_length:seq_length + len(val_predictions)]
                val_actual_all = y_val[seq_length:seq_length + len(val_predictions)]
                
                # Apply the same open mask to get only open ride times
                if 'closed' in val_eval_df.columns:
                    val_linear_open = val_linear_all[open_mask]
                    val_actual_open = val_actual_all[open_mask]
                else:
                    val_linear_open = val_linear_all
                    val_actual_open = val_actual_all
                
                # Combined predictions = linear baseline + TCN residual predictions
                val_combined_open = val_linear_open + val_predictions_open
                
                # Calculate sMAPE on actual wait times vs combined predictions
                # Filter out zero wait times to avoid sMAPE = 200% for those cases
                non_zero_mask = val_actual_open > 0.1  # Only consider wait times > 0.1 minutes
                
                if np.sum(non_zero_mask) > 0:
                    val_actual_nonzero = val_actual_open[non_zero_mask]
                    val_combined_nonzero = val_combined_open[non_zero_mask]
                    
                    # sMAPE = (|actual - predicted|) / (|actual| + |predicted|) * 100
                    numerator = np.abs(val_actual_nonzero - val_combined_nonzero)
                    denominator = np.abs(val_actual_nonzero) + np.abs(val_combined_nonzero)
                    val_smape = np.mean(numerator / denominator) * 100

                else:
                    val_smape = 0.0  # All wait times are zero
            else:
                # Fallback if no open samples (shouldn't happen normally)
                val_mae = val_rmse = val_smape = float('inf')
            
            # Step scheduler and get current learning rate
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            # Logging
            metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mae": val_mae,
                "val_rmse": val_rmse,
                "val_smape": val_smape,
                "learning_rate": current_lr
            }
            
            if wandb.run:
                wandb.log(metrics)
            
            logger.info(f'Epoch {epoch+1}/{self.config["epochs"]} - Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}, Val sMAPE: {val_smape:.2f}%, LR: {current_lr:.6f}')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = tcn_model.state_dict().copy()
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        if best_model is not None:
            tcn_model.load_state_dict(best_model)
        
        # Move model to CPU for evaluation to free GPU memory
        tcn_model = tcn_model.cpu()
        
        # Clear GPU cache after moving model to CPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Evaluation
        logger.info("Evaluating model...")
        tcn_model.eval()
        
        # Get TCN predictions
        all_tcn_preds = []
        with torch.no_grad():
            for inputs, _ in test_loader:
                outputs = tcn_model(inputs)
                all_tcn_preds.extend(outputs.numpy().flatten())
        
        # Calculate metrics
        y_test_seq_linear = y_test_pred_linear[seq_length:][:len(all_tcn_preds)]
        y_test_seq_actual = y_test[seq_length:][:len(all_tcn_preds)]
        
        test_eval_df = test_df.iloc[seq_length:].reset_index(drop=True).iloc[:len(all_tcn_preds)].copy()
        test_eval_df['linear_pred'] = y_test_seq_linear
        test_eval_df['tcn_pred'] = all_tcn_preds
        test_eval_df['combined_pred'] = y_test_seq_linear + np.array(all_tcn_preds)
        
        # Filter out closed rides
        if 'closed' in test_eval_df.columns:
            logger.info(f"Excluding {test_eval_df['closed'].sum()} data points where ride is closed")
            open_ride_df = test_eval_df[test_eval_df['closed'] == 0]
        else:
            logger.warning("'closed' column not found. Evaluating on all test data.")
            open_ride_df = test_eval_df
        
        # Calculate final metrics
        y_test_open_actual = open_ride_df['wait_time'].values
        y_test_open_linear = open_ride_df['linear_pred'].values
        y_test_open_combined = open_ride_df['combined_pred'].values

        combined_mae = mean_absolute_error(y_test_open_actual, y_test_open_combined)
        combined_rmse = np.sqrt(mean_squared_error(y_test_open_actual, y_test_open_combined))
        combined_r2 = r2_score(y_test_open_actual, y_test_open_combined)
        non_zero_mask = y_test_open_actual > 0.1  # Only consider wait times > 0.1 minutes

        if np.sum(non_zero_mask) > 0:
            y_test_actual_nonzero = y_test_open_actual[non_zero_mask]
            y_test_combined_nonzero = y_test_open_combined[non_zero_mask]
            
            # sMAPE = (|actual - predicted|) / (|actual| + |predicted|) * 100
            numerator = np.abs(y_test_actual_nonzero - y_test_combined_nonzero)
            denominator = np.abs(y_test_actual_nonzero) + np.abs(y_test_combined_nonzero)
            combined_smape = np.mean(numerator / denominator) * 100
        else:
            combined_smape = 0.0  # All wait times are zero

        final_metrics = {
            "combined_mae": combined_mae,
            "combined_rmse": combined_rmse,
            "combined_r2": combined_r2,
            "combined_smape": combined_smape,  # Add this line
            "best_val_loss": best_val_loss,
        }        

        
        if wandb.run:
            wandb.log(final_metrics)
        
        logger.info(f"Final metrics: {final_metrics}")
        
        # Save models
        self._save_models(linear_model, tcn_model)
        
        # AGGRESSIVE cleanup - delete everything and clear all references
        del best_model
        del tcn_model, linear_model, optimizer, scheduler, criterion
        del train_dataset, val_dataset, test_dataset
        del train_loader, val_loader, test_loader
        del X_train, X_val, X_test, y_train, y_val, y_test
        del y_train_pred_linear, y_val_pred_linear, y_test_pred_linear
        del train_residuals, val_residuals, test_residuals
        del train_df, val_df, test_df, df
        del test_eval_df, open_ride_df
        del all_tcn_preds
        
        # Force multiple garbage collections
        import gc
        for _ in range(3):
            gc.collect()
        
        # Clear GPU cache multiple times
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        return final_metrics
    
    def _save_models(self, linear_model, tcn_model):
        """Save trained models"""
        os.makedirs("models", exist_ok=True)
        target_ride = self.config['target_ride'].replace(' ', '_')
        
        # Save linear model
        linear_model_filename = f"{target_ride}_linear_model.pkl"
        with open(f"models/{linear_model_filename}", "wb") as f:
            pickle.dump(linear_model, f)
        
        # Save TCN model
        tcn_model_filename = f"{target_ride}_tcn_model.pt"
        torch.save(tcn_model.state_dict(), f"models/{tcn_model_filename}")
        
        logger.info(f"Models saved: {linear_model_filename}, {tcn_model_filename}")
        
        # Log to wandb if available
        if wandb.run:
            linear_artifact = wandb.Artifact(f"linear_model_{wandb.run.id}", type="model")
            linear_artifact.add_file(f"models/{linear_model_filename}")
            wandb.log_artifact(linear_artifact)
            
            tcn_artifact = wandb.Artifact(f"tcn_model_{wandb.run.id}", type="model")
            tcn_artifact.add_file(f"models/{tcn_model_filename}")
            wandb.log_artifact(tcn_artifact)

def load_config(config_path):
    """Load configuration from YAML or JSON file"""
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            return yaml.safe_load(f)
        else:
            return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Train TCN models for wait time prediction")
    parser.add_argument('--config', help='Path to configuration file (optional if using --ride)')
    parser.add_argument('--ride', help='Ride name (alternative to --config)')
    parser.add_argument('--rides-config', default='configs/rides_config.yaml',
                       help='Path to rides configuration file')
    parser.add_argument('--mode', choices=['single', 'sweep'], default='single', 
                       help='Training mode: single model or hyperparameter sweep')
    parser.add_argument('--wandb-project', default='waitless-tcn-hslu-dspro2-fs25',
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
        logger.info(f"Created configuration for ride: {args.ride}")
    else:
        raise ValueError("Either --config or --ride must be specified")
    
    if args.mode == 'single':
        # Single model training
        if config.get('use_wandb', True):
            run_name = f"{config['target_ride']}_single"
            if config.get('run_name'):
                run_name = f"{config['target_ride']}_{config['run_name']}"
            
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                config=config,
                name=run_name
            )
        
        trainer = TCNTrainer(config)
        metrics = trainer.train_single_model()
        
        if wandb.run:
            wandb.finish()
            
    elif args.mode == 'sweep':
        # Hyperparameter sweep
        logger.info("Starting hyperparameter sweep...")
        # This will be handled by the sweep configuration
        trainer = TCNTrainer(config)
        trainer.train_single_model()

def create_config_from_ride(ride_name, rides_config_path="configs/rides_config.yaml"):
    """Create a training config from ride name and rides config"""
    with open(rides_config_path, 'r') as f:
        rides_config = yaml.safe_load(f)
    
    if ride_name not in rides_config['rides']:
        raise ValueError(f"Ride '{ride_name}' not found in {rides_config_path}. Available rides: {list(rides_config['rides'].keys())}")
    
    # Start with default parameters
    config = rides_config['default_params'].copy()
    
    # Add global settings
    config.update(rides_config['global_settings'])
    
    # Add ride-specific settings
    ride_info = rides_config['rides'][ride_name]
    config['data_path'] = ride_info['data_path']
    config['target_ride'] = ride_name
    
    # Add some default training hyperparameters if not specified
    default_hyperparams = {
        'seq_length': 96,
        'batch_size': 256,
        'num_channels': 128,
        'kernel_size': 3,
        'dropout': 0.2,
        'learning_rate': 0.001,
        'scheduler_type': 'CosineAnnealingLR',
        't_max': 50,
        'eta_min': 1e-6,
        'run_name': 'default_config'
    }
    
    for key, value in default_hyperparams.items():
        if key not in config:
            config[key] = value
    
    return config

if __name__ == "__main__":
    main()