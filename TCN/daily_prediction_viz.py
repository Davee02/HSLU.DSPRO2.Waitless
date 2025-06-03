#!/usr/bin/env python3
"""
Daily Prediction Visualization Script
Predicts a whole day and creates beautiful visualizations comparing actual vs predicted values.
Works with both original TCN and autoregressive models.
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pytorch_tcn import TCN
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
import pickle
import yaml

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('daily_prediction.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class OriginalTCNModel(nn.Module):
    """Original TCN model for residual prediction"""
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout=0.2):
        super(OriginalTCNModel, self).__init__()
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

class AutoregressiveTCNModel(nn.Module):
    """Autoregressive TCN model"""
    def __init__(self, static_features_size, seq_length, output_size, 
                 num_channels, kernel_size, dropout=0.2, num_layers=8):
        super(AutoregressiveTCNModel, self).__init__()
        
        self.static_features_size = static_features_size
        self.seq_length = seq_length
        self.autoregressive_size = seq_length * 2
        self.tcn_input_size = static_features_size + 2
        
        channels = [num_channels] * num_layers
        
        self.tcn = TCN(
            num_inputs=self.tcn_input_size,
            num_channels=channels,
            kernel_size=kernel_size,
            dropout=dropout,
            causal=True,
            use_skip_connections=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(num_channels, num_channels // 2)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(num_channels // 2, output_size)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        static_features = x[:, :self.static_features_size]
        autoregressive_features = x[:, self.static_features_size:]
        
        autoregressive_reshaped = autoregressive_features.view(
            batch_size, self.seq_length, 2
        )
        
        static_repeated = static_features.unsqueeze(1).repeat(
            1, self.seq_length, 1
        )
        
        combined_sequence = torch.cat([
            static_repeated, autoregressive_reshaped
        ], dim=2)
        
        tcn_input = combined_sequence.transpose(1, 2)
        tcn_out = self.tcn(tcn_input)
        last_hidden = tcn_out[:, :, -1]
        
        out = self.dropout(last_hidden)
        out = self.linear1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.linear2(out)
        
        return out

class DailyPredictor:
    def __init__(self, ride_name, model_type='auto', device=None):
        self.ride_name = ride_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ride_safe_name = ride_name.replace(' ', '_')
        
        # Detect and load models
        self.model_type = self._detect_model_type() if model_type == 'auto' else model_type
        logger.info(f"Using {self.model_type} model approach")
        
        if self.model_type == 'autoregressive':
            self._load_autoregressive_models()
        else:
            self._load_original_models()
    
    def _detect_model_type(self):
        """Auto-detect which model type is available"""
        # Check for autoregressive models first
        gb_path = f"models/autoregressive/{self.ride_safe_name}_gb_baseline.pkl"
        tcn_ar_path = f"models/autoregressive/{self.ride_safe_name}_autoregressive_tcn.pt"
        
        if os.path.exists(gb_path) and os.path.exists(tcn_ar_path):
            return 'autoregressive'
        
        # Check for original models
        linear_path = f"models/{self.ride_safe_name}_linear_model.pkl"
        tcn_path = f"models/{self.ride_safe_name}_tcn_model.pt"
        
        if os.path.exists(linear_path) and os.path.exists(tcn_path):
            return 'original'
        
        # Check for best_model.pt (might be renamed)
        if os.path.exists("models/best_model.pt"):
            logger.warning("Found best_model.pt but cannot determine type. Assuming original.")
            return 'original'
        
        raise FileNotFoundError(f"No trained models found for ride {self.ride_name}")
    
    def _load_autoregressive_models(self):
        """Load autoregressive models"""
        gb_path = f"models/autoregressive/{self.ride_safe_name}_gb_baseline.pkl"
        tcn_path = f"models/autoregressive/{self.ride_safe_name}_autoregressive_tcn.pt"
        
        # Load GradientBoosting
        with open(gb_path, 'rb') as f:
            self.gb_model = pickle.load(f)
        
        # Load TCN
        checkpoint = torch.load(tcn_path, map_location=self.device)
        self.config = checkpoint['config']
        
        self.static_features_size = self.gb_model.n_features_in_
        self.seq_length = self.config['seq_length']
        self.opening_hour = self.config.get('opening_hour', 9)
        self.closing_hour = self.config.get('closing_hour', 21)
        
        self.tcn_model = AutoregressiveTCNModel(
            static_features_size=self.static_features_size,
            seq_length=self.seq_length,
            output_size=1,
            num_channels=self.config['num_channels'],
            kernel_size=self.config['kernel_size'],
            dropout=self.config['dropout'],
            num_layers=self.config.get('num_layers', 8)
        )
        
        # Handle compiled model state dict (removes _orig_mod. prefix)
        state_dict = checkpoint['model_state_dict']
        if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
            logger.info("Detected compiled model, removing _orig_mod. prefix")
            state_dict = {key.replace('_orig_mod.', ''): value for key, value in state_dict.items()}
        
        self.tcn_model.load_state_dict(state_dict)
        self.tcn_model.to(self.device)
        self.tcn_model.eval()
        
        logger.info(f"Loaded autoregressive models (seq_length: {self.seq_length})")
    
    def _load_original_models(self):
        """Load original TCN models"""
        # Try different possible paths
        model_paths = [
            f"models/{self.ride_safe_name}_linear_model.pkl",
            f"models/{self.ride_safe_name}_gb_baseline.pkl",  # Fallback
            "models/best_linear_model.pkl"
        ]
        
        tcn_paths = [
            f"models/{self.ride_safe_name}_tcn_model.pt",
            "models/best_model.pt"
        ]
        
        # Load baseline model
        baseline_path = None
        for path in model_paths:
            if os.path.exists(path):
                baseline_path = path
                break
        
        if baseline_path is None:
            raise FileNotFoundError("No baseline model found")
        
        with open(baseline_path, 'rb') as f:
            self.gb_model = pickle.load(f)
        
        # Load TCN model
        tcn_path = None
        for path in tcn_paths:
            if os.path.exists(path):
                tcn_path = path
                break
        
        if tcn_path is None:
            raise FileNotFoundError("No TCN model found")
        
        # Try to load as state dict or full model
        try:
            if tcn_path.endswith('best_model.pt'):
                # Might be a full model or state dict
                model_data = torch.load(tcn_path, map_location=self.device)
                if isinstance(model_data, dict) and 'model_state_dict' in model_data:
                    model_data = model_data['model_state_dict']
            else:
                model_data = torch.load(tcn_path, map_location=self.device)
            
            # Create model with reasonable defaults
            self.seq_length = 96  # Default sequence length
            self.opening_hour = 9
            self.closing_hour = 21
            
            # Determine input size from GB model
            input_size = self.gb_model.n_features_in_
            
            self.tcn_model = OriginalTCNModel(
                input_size=input_size,
                output_size=1,
                num_channels=128,  # Default
                kernel_size=3,
                dropout=0.2
            )
            
            self.tcn_model.load_state_dict(model_data)
            self.tcn_model.to(self.device)
            self.tcn_model.eval()
            
        except Exception as e:
            logger.error(f"Error loading TCN model: {e}")
            raise
        
        logger.info(f"Loaded original models (seq_length: {self.seq_length})")
    
    def preprocess_data(self, df):
        """Preprocess data for prediction"""
        logger.info(f"Preprocessing data for {self.ride_name}")
        
        # Remove time_bucket if present
        if 'time_bucket' in df.columns:
            df = df.drop(columns=['time_bucket'])
        
        # Filter for target ride
        ride_col = f'ride_name_{self.ride_name}'
        if ride_col in df.columns:
            df = df[df[ride_col] == 1].copy()
        
        # Remove all ride_name columns
        ride_cols = [col for col in df.columns if col.startswith('ride_name_')]
        df = df.drop(columns=ride_cols)
        
        # Handle missing values
        df = df.fillna(0)
        
        # Ensure timestamp is datetime and sort
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Get feature columns
        if self.model_type == 'autoregressive':
            static_feature_cols = [col for col in df.columns 
                                  if col not in ['wait_time', 'timestamp']]
        else:
            static_feature_cols = [col for col in df.columns 
                                  if col not in ['wait_time', 'timestamp']]
        
        return df, static_feature_cols
    
    def predict_day(self, df, feature_cols, target_date):
        """Predict a full day"""
        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date).date()
        elif isinstance(target_date, pd.Timestamp):
            target_date = target_date.date()
        elif isinstance(target_date, datetime):
            target_date = target_date.date()
        
        logger.info(f"Predicting full day: {target_date}")
        
        # Get actual data for the target date
        df['date'] = df['timestamp'].dt.date
        day_mask = df['date'] == target_date
        actual_day_data = df[day_mask].copy()
        
        if len(actual_day_data) == 0:
            raise ValueError(f"No actual data found for {target_date}")
        
        # Generate prediction timestamps for operating hours
        prediction_timestamps = []
        base_datetime = datetime.combine(target_date, datetime.min.time())
        
        for hour in range(self.opening_hour, self.closing_hour + 1):
            for minute in [0, 15, 30, 45]:
                timestamp = base_datetime.replace(hour=hour, minute=minute)
                prediction_timestamps.append(timestamp)
        
        predictions = []
        
        if self.model_type == 'autoregressive':
            predictions = self._predict_autoregressive(df, feature_cols, prediction_timestamps)
        else:
            predictions = self._predict_original(df, feature_cols, prediction_timestamps)
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'timestamp': prediction_timestamps,
            'predicted_wait_time': predictions
        })
        
        # Merge with actual data
        combined_df = results_df.merge(
            actual_day_data[['timestamp', 'wait_time']],
            on='timestamp',
            how='outer',
            suffixes=('_pred', '_actual')
        ).sort_values('timestamp')
        
        # Fill missing predictions with interpolation
        combined_df['predicted_wait_time'] = combined_df['predicted_wait_time'].interpolate()
        
        return combined_df, actual_day_data
    
    def _predict_autoregressive(self, df, feature_cols, timestamps):
        """Predict using autoregressive model"""
        predictions = []
        df_work = df.copy()
        
        # Pre-calculate historical GB predictions and residuals
        historical_gb_preds = self.gb_model.predict(df_work[feature_cols].values)
        historical_residuals = df_work['wait_time'].values - historical_gb_preds
        df_work['gb_pred'] = historical_gb_preds
        df_work['residual'] = historical_residuals
        
        for timestamp in timestamps:
            # Get static features
            static_features = self._get_static_features_for_timestamp(df_work, feature_cols, timestamp)
            if static_features is None:
                predictions.append(0)
                continue
            
            # GB prediction
            gb_pred = self.gb_model.predict([static_features])[0]
            
            # Get autoregressive sequence
            autoregressive_seq = self._get_autoregressive_sequence(df_work, static_features, timestamp)
            if autoregressive_seq is None:
                predictions.append(max(0, gb_pred))
                continue
            
            # TCN residual prediction
            with torch.no_grad():
                seq_tensor = torch.FloatTensor(autoregressive_seq).unsqueeze(0).to(self.device)
                residual_pred = self.tcn_model(seq_tensor).cpu().numpy()[0, 0]
            
            # Combined prediction
            final_pred = gb_pred + residual_pred
            final_pred = max(0, final_pred)
            predictions.append(final_pred)
            
            # Update dataframe
            self._update_dataframe_simple(df_work, timestamp, final_pred, gb_pred, residual_pred)
        
        return predictions
    
    def _predict_original(self, df, feature_cols, timestamps):
        """Predict using original model"""
        predictions = []
        
        # Get baseline predictions for all data
        all_gb_preds = self.gb_model.predict(df[feature_cols].values)
        
        for i, timestamp in enumerate(timestamps):
            # Find corresponding data point or interpolate
            timestamp_mask = df['timestamp'] <= timestamp
            if not timestamp_mask.any():
                predictions.append(0)
                continue
            
            # Get the most recent seq_length points
            recent_indices = df[timestamp_mask].tail(self.seq_length).index
            
            if len(recent_indices) < self.seq_length:
                # Not enough historical data, use GB prediction only
                closest_idx = df[timestamp_mask].index[-1]
                predictions.append(max(0, all_gb_preds[closest_idx]))
                continue
            
            # Create sequence
            X_seq = df.loc[recent_indices, feature_cols].values
            
            # Get residuals for the sequence
            gb_preds_seq = all_gb_preds[recent_indices]
            actual_seq = df.loc[recent_indices, 'wait_time'].values
            residuals_seq = actual_seq - gb_preds_seq
            
            # TCN prediction
            with torch.no_grad():
                seq_tensor = torch.FloatTensor(X_seq).unsqueeze(0).to(self.device)
                residual_pred = self.tcn_model(seq_tensor).cpu().numpy()[0, 0]
            
            # Get GB prediction for current timestamp
            closest_idx = df[timestamp_mask].index[-1]
            gb_pred = all_gb_preds[closest_idx]
            
            # Combined prediction
            final_pred = gb_pred + residual_pred
            predictions.append(max(0, final_pred))
        
        return predictions
    
    def _get_static_features_for_timestamp(self, df, feature_cols, timestamp):
        """Get static features for a timestamp"""
        # Try exact match first
        exact_match = df[df['timestamp'] == timestamp]
        if not exact_match.empty:
            return exact_match[feature_cols].iloc[0].values
        
        # Use most recent and update time features
        latest_row = df.iloc[-1][feature_cols].copy()
        
        # Update time-based features if they exist
        if 'hour' in feature_cols:
            latest_row['hour'] = timestamp.hour
        if 'minute' in feature_cols:
            latest_row['minute'] = timestamp.minute
        if 'day_of_week' in feature_cols:
            latest_row['day_of_week'] = timestamp.weekday()
        if 'is_weekend' in feature_cols:
            latest_row['is_weekend'] = int(timestamp.weekday() >= 5)
        
        return latest_row.values
    
    def _get_autoregressive_sequence(self, df, static_features, timestamp):
        """Get autoregressive sequence for TCN"""
        mask_before = df['timestamp'] < timestamp
        if not mask_before.any():
            return None
        
        recent_data = df[mask_before].tail(self.seq_length)
        if len(recent_data) < self.seq_length:
            return None
        
        # Extract autoregressive features
        autoregressive_features = []
        for _, row in recent_data.iterrows():
            wait_time = row.get('wait_time', 0)
            residual = row.get('residual', 0)
            autoregressive_features.extend([wait_time, residual])
        
        # Combine features
        combined_features = np.concatenate([static_features, autoregressive_features])
        return combined_features
    
    def _update_dataframe_simple(self, df, timestamp, wait_time, gb_pred, residual):
        """Simple dataframe update for predictions"""
        # Check if timestamp exists
        existing_mask = df['timestamp'] == timestamp
        
        if existing_mask.any():
            df.loc[existing_mask, 'wait_time'] = wait_time
            df.loc[existing_mask, 'gb_pred'] = gb_pred
            df.loc[existing_mask, 'residual'] = residual
        else:
            # Add new row (simplified)
            new_row = df.iloc[-1].copy()
            new_row['timestamp'] = timestamp
            new_row['wait_time'] = wait_time
            new_row['gb_pred'] = gb_pred
            new_row['residual'] = residual
            
            df.loc[len(df)] = new_row
    
    def calculate_metrics(self, combined_df):
        """Calculate prediction metrics"""
        # Remove rows where either actual or predicted is missing
        valid_data = combined_df.dropna(subset=['wait_time', 'predicted_wait_time'])
        
        if len(valid_data) == 0:
            return None
        
        y_true = valid_data['wait_time'].values
        y_pred = valid_data['predicted_wait_time'].values
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # sMAPE
        non_zero_mask = y_true > 0.1
        if np.sum(non_zero_mask) > 0:
            y_true_nz = y_true[non_zero_mask]
            y_pred_nz = y_pred[non_zero_mask]
            numerator = np.abs(y_true_nz - y_pred_nz)
            denominator = np.abs(y_true_nz) + np.abs(y_pred_nz)
            smape = np.mean(numerator / denominator) * 100
        else:
            smape = 0.0
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'smape': smape,
            'num_samples': len(valid_data)
        }
    
    def create_visualization(self, combined_df, metrics, save_path=None):
        """Create beautiful visualization"""
        # Set up the plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Daily Wait Time Prediction - {self.ride_name}\n{combined_df["timestamp"].dt.date.iloc[0]}', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Time series comparison
        ax1 = axes[0, 0]
        
        # Plot actual data
        actual_data = combined_df.dropna(subset=['wait_time'])
        if len(actual_data) > 0:
            ax1.plot(actual_data['timestamp'], actual_data['wait_time'], 
                    'o-', color='red', linewidth=2, markersize=4, 
                    label='Actual', alpha=0.8)
        
        # Plot predictions
        pred_data = combined_df.dropna(subset=['predicted_wait_time'])
        if len(pred_data) > 0:
            ax1.plot(pred_data['timestamp'], pred_data['predicted_wait_time'], 
                    's-', color='blue', linewidth=2, markersize=4, 
                    label='Predicted', alpha=0.8)
        
        ax1.set_title('Wait Time Predictions vs Actual', fontweight='bold')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Wait Time (minutes)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Scatter plot
        ax2 = axes[0, 1]
        
        valid_data = combined_df.dropna(subset=['wait_time', 'predicted_wait_time'])
        if len(valid_data) > 0:
            ax2.scatter(valid_data['wait_time'], valid_data['predicted_wait_time'], 
                       alpha=0.6, s=50, color='purple')
            
            # Perfect prediction line
            max_val = max(valid_data['wait_time'].max(), valid_data['predicted_wait_time'].max())
            ax2.plot([0, max_val], [0, max_val], 'r--', alpha=0.7, label='Perfect Prediction')
            
            ax2.set_xlabel('Actual Wait Time (minutes)')
            ax2.set_ylabel('Predicted Wait Time (minutes)')
            ax2.set_title('Prediction Accuracy Scatter Plot', fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Residuals
        ax3 = axes[1, 0]
        
        if len(valid_data) > 0:
            residuals = valid_data['predicted_wait_time'] - valid_data['wait_time']
            ax3.scatter(valid_data['timestamp'], residuals, alpha=0.6, s=30, color='green')
            ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            ax3.set_title('Prediction Residuals Over Time', fontweight='bold')
            ax3.set_xlabel('Time')
            ax3.set_ylabel('Residual (Predicted - Actual)')
            ax3.grid(True, alpha=0.3)
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Metrics summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        if metrics:
            metrics_text = f"""
MODEL PERFORMANCE METRICS

MAE (Mean Absolute Error): {metrics['mae']:.2f} min
RMSE (Root Mean Square Error): {metrics['rmse']:.2f} min
R² (Coefficient of Determination): {metrics['r2']:.3f}
sMAPE (Symmetric MAPE): {metrics['smape']:.1f}%

Samples: {metrics['num_samples']}
Model Type: {self.model_type.title()}

Interpretation:
• MAE: Average prediction error
• RMSE: Penalizes larger errors more
• R²: How well model explains variance (1.0 = perfect)
• sMAPE: Symmetric percentage error
            """
            
            ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes, 
                    fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        plt.show()

def load_rides_config(config_path):
    """Load rides configuration"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Rides config not found: {config_path}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Daily prediction visualization")
    parser.add_argument('--ride', required=True, help='Ride name for prediction')
    parser.add_argument('--data-path', required=True, help='Path to data file')
    parser.add_argument('--date', help='Target date (YYYY-MM-DD), defaults to latest available')
    parser.add_argument('--model-type', choices=['original', 'autoregressive', 'auto'], 
                       default='auto', help='Model type to use')
    parser.add_argument('--save-path', help='Path to save visualization')
    parser.add_argument('--rides-config', default='configs/rides_config.yaml',
                       help='Path to rides configuration file')
    
    args = parser.parse_args()
    
    try:
        # Load predictor
        predictor = DailyPredictor(args.ride, args.model_type)
        
        # Load and preprocess data
        df = pd.read_parquet(args.data_path)
        df, feature_cols = predictor.preprocess_data(df)
        
        # Determine target date
        if args.date:
            target_date = pd.to_datetime(args.date).date()
        else:
            # Use the most recent date with data
            target_date = df['timestamp'].dt.date.max()
        
        logger.info(f"Predicting for date: {target_date}")
        
        # Make predictions
        combined_df, actual_day_data = predictor.predict_day(df, feature_cols, target_date)
        
        # Calculate metrics
        metrics = predictor.calculate_metrics(combined_df)
        
        # Print metrics
        if metrics:
            print(f"\n🎯 PREDICTION METRICS for {args.ride} on {target_date}")
            print("=" * 60)
            print(f"MAE (Mean Absolute Error):    {metrics['mae']:.2f} minutes")
            print(f"RMSE (Root Mean Square Error): {metrics['rmse']:.2f} minutes")
            print(f"R² (Coefficient of Determination): {metrics['r2']:.3f}")
            print(f"sMAPE (Symmetric MAPE):       {metrics['smape']:.1f}%")
            print(f"Number of samples:            {metrics['num_samples']}")
            print(f"Model type:                   {predictor.model_type.title()}")
        
        # Create visualization
        save_path = args.save_path or f"predictions/{args.ride}_{target_date}_daily_prediction.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        predictor.create_visualization(combined_df, metrics, save_path)
        
        # Save data
        data_save_path = save_path.replace('.png', '_data.csv')
        combined_df.to_csv(data_save_path, index=False)
        logger.info(f"Prediction data saved to {data_save_path}")
        
        print(f"\n✅ Daily prediction completed!")
        print(f"📊 Visualization saved: {save_path}")
        print(f"💾 Data saved: {data_save_path}")
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise

if __name__ == "__main__":
    main()