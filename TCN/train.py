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


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
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
        df = pd.read_parquet(self.config['data_path'])
        df = self.preprocess_data(df, self.config['target_ride'])

        train_indices, val_indices, test_indices = self.load_data_splits(
            self.config['splits_output_dir'], 
            self.config['target_ride']
        )
        
        df, feature_cols = self.create_features(df)

        train_df = df.iloc[train_indices].copy()
        val_df = df.iloc[val_indices].copy()
        test_df = df.iloc[test_indices].copy()

        X_train = train_df[feature_cols].values
        y_train = train_df['wait_time'].values
        X_val = val_df[feature_cols].values
        y_val = val_df['wait_time'].values
        X_test = test_df[feature_cols].values
        y_test = test_df['wait_time'].values

        logger.info("Training linear model...")
        linear_model = LinearRegression()
        linear_model.fit(X_train, y_train)
        
        y_train_pred_linear = linear_model.predict(X_train)
        y_val_pred_linear = linear_model.predict(X_val)
        y_test_pred_linear = linear_model.predict(X_test)
        
        train_residuals = y_train - y_train_pred_linear
        val_residuals = y_val - y_val_pred_linear
        test_residuals = y_test - y_test_pred_linear
        
        seq_length = self.config['seq_length']
        train_dataset = TimeSeriesDataset(X_train, train_residuals, seq_length)
        val_dataset = TimeSeriesDataset(X_val, val_residuals, seq_length)
        test_dataset = TimeSeriesDataset(X_test, test_residuals, seq_length)
        
        batch_size = self.config['batch_size']
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        logger.info("Initializing TCN model...")
        input_size = X_train.shape[1]
        tcn_model = TCNModel(
            input_size=input_size,
            output_size=1,
            num_channels=self.config['num_channels'],
            kernel_size=self.config['kernel_size'],
            dropout=self.config['dropout']
        ).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(tcn_model.parameters(), lr=self.config['learning_rate'])
        
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.config.get('t_max', self.config['epochs']),
            eta_min=self.config.get('eta_min', 1e-6)
        )
        
        logger.info("Starting TCN training...")
        best_val_loss = float('inf')
        counter = 0
        best_model = None
        patience = self.config.get('patience', 10)
        
        for epoch in range(self.config['epochs']):
            tcn_model.train()
            train_loss = 0.0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = tcn_model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            train_loss /= len(train_loader)
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            tcn_model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = tcn_model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    
            val_loss /= len(val_loader)

            metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch": epoch,
                "learning_rate": current_lr
            }
            
            if wandb.run:
                wandb.log(metrics)
            
            logger.info(f'Epoch {epoch+1}/{self.config["epochs"]} - Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, LR: {current_lr:.6f}')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = tcn_model.state_dict().copy()
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        if best_model is not None:
            tcn_model.load_state_dict(best_model)

        logger.info("Evaluating model...")
        tcn_model.to(torch.device("cpu"))
        tcn_model.eval()

        all_tcn_preds = []
        with torch.no_grad():
            for inputs, _ in test_loader:
                outputs = tcn_model(inputs)
                all_tcn_preds.extend(outputs.numpy().flatten())
        
        y_test_seq_linear = y_test_pred_linear[seq_length:][:len(all_tcn_preds)]
        y_test_seq_actual = y_test[seq_length:][:len(all_tcn_preds)]
        
        test_eval_df = test_df.iloc[seq_length:].reset_index(drop=True).iloc[:len(all_tcn_preds)].copy()
        test_eval_df['linear_pred'] = y_test_seq_linear
        test_eval_df['tcn_pred'] = all_tcn_preds
        test_eval_df['combined_pred'] = y_test_seq_linear + np.array(all_tcn_preds)
        

        if 'closed' in test_eval_df.columns:
            logger.info(f"Excluding {test_eval_df['closed'].sum()} data points where ride is closed")
            open_ride_df = test_eval_df[test_eval_df['closed'] == 0]
        else:
            logger.warning("'closed' column not found. Evaluating on all test data.")
            open_ride_df = test_eval_df

        y_test_open_actual = open_ride_df['wait_time'].values
        y_test_open_linear = open_ride_df['linear_pred'].values
        y_test_open_combined = open_ride_df['combined_pred'].values
        
        linear_mae = mean_absolute_error(y_test_open_actual, y_test_open_linear)
        combined_mae = mean_absolute_error(y_test_open_actual, y_test_open_combined)
        combined_rmse = np.sqrt(mean_squared_error(y_test_open_actual, y_test_open_combined))
        combined_r2 = r2_score(y_test_open_actual, y_test_open_combined)
        
        final_metrics = {
            "linear_mae": linear_mae,
            "combined_mae": combined_mae,
            "combined_rmse": combined_rmse,
            "combined_r2": combined_r2,
            "best_val_loss": best_val_loss,
        }
        
        if wandb.run:
            wandb.log(final_metrics)
        
        logger.info(f"Final metrics: {final_metrics}")

        self._save_models(linear_model, tcn_model)
        
        return final_metrics
    
    def _save_models(self, linear_model, tcn_model):
        """Save trained models"""
        os.makedirs("models", exist_ok=True)
        target_ride = self.config['target_ride'].replace(' ', '_')
        
        linear_model_filename = f"{target_ride}_linear_model.pkl"
        with open(f"models/{linear_model_filename}", "wb") as f:
            pickle.dump(linear_model, f)

        tcn_model_filename = f"{target_ride}_tcn_model.pt"
        torch.save(tcn_model.state_dict(), f"models/{tcn_model_filename}")
        
        logger.info(f"Models saved: {linear_model_filename}, {tcn_model_filename}")

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

    if args.config:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    elif args.ride:
        config = create_config_from_ride(args.ride, args.rides_config)
        logger.info(f"Created configuration for ride: {args.ride}")
    else:
        raise ValueError("Either --config or --ride must be specified")
    
    if args.mode == 'single':

        if config.get('use_wandb', True):
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                config=config,
                name=f"single_{config['target_ride']}_{config.get('run_name', '')}"
            )
        
        trainer = TCNTrainer(config)
        metrics = trainer.train_single_model()
        
        if wandb.run:
            wandb.finish()
            
    elif args.mode == 'sweep':

        logger.info("Starting hyperparameter sweep...")

        trainer = TCNTrainer(config)
        trainer.train_single_model()

def create_config_from_ride(ride_name, rides_config_path="configs/rides_config.yaml"):
    """Create a training config from ride name and rides config"""
    with open(rides_config_path, 'r') as f:
        rides_config = yaml.safe_load(f)
    
    if ride_name not in rides_config['rides']:
        raise ValueError(f"Ride '{ride_name}' not found in {rides_config_path}. Available rides: {list(rides_config['rides'].keys())}")
    
    config = rides_config['default_params'].copy()
    
    config.update(rides_config['global_settings'])
    
    ride_info = rides_config['rides'][ride_name]
    config['data_path'] = ride_info['data_path']
    config['target_ride'] = ride_name

    default_hyperparams = {
        'seq_length': 96,
        'batch_size': 256,
        'num_channels': 128,
        'kernel_size': 3,
        'dropout': 0.2,
        'learning_rate': 0.0001,
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
