#!/usr/bin/env python3
"""
Wait Time Prediction Inference Script
Loads trained GradientBoosting + TCN models to predict wait times N days into the future
with comprehensive visualizations and uncertainty estimation.
"""

import argparse
import json
import logging
import os
import pickle
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import torch
import torch.nn as nn
from pytorch_tcn import TCN
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class AutoregressiveTCNModel(nn.Module):
    """
    TCN model for autoregressive residual prediction.
    Matches the architecture from the training script.
    """
    def __init__(self, static_features_size, seq_length, output_size, 
                 num_channels, kernel_size, dropout=0.2, num_layers=8):
        super(AutoregressiveTCNModel, self).__init__()
        
        self.static_features_size = static_features_size
        self.seq_length = seq_length
        self.autoregressive_size = seq_length * 2  # wait_time + residual per timestep
        self.total_input_size = static_features_size + self.autoregressive_size
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
        
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(num_channels, num_channels // 2)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(num_channels // 2, output_size)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Split input into static and autoregressive parts
        static_features = x[:, :self.static_features_size]
        autoregressive_features = x[:, self.static_features_size:]
        
        # Reshape autoregressive features
        autoregressive_reshaped = autoregressive_features.view(
            batch_size, self.seq_length, 2
        )
        
        # Repeat static features for each timestep
        static_repeated = static_features.unsqueeze(1).repeat(
            1, self.seq_length, 1
        )
        
        # Combine static and autoregressive features
        combined_sequence = torch.cat([
            static_repeated, autoregressive_reshaped
        ], dim=2)
        
        # TCN expects (batch_size, features, seq_length)
        tcn_input = combined_sequence.transpose(1, 2)
        
        # TCN forward pass
        tcn_out = self.tcn(tcn_input)
        
        # Take the last time step
        last_hidden = tcn_out[:, :, -1]
        
        # Output layers
        out = self.dropout(last_hidden)
        out = self.linear1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.linear2(out)
        
        return out


class WaitTimePredictionEngine:
    """
    Engine for predicting wait times using trained GradientBoosting + TCN models.
    """
    
    def __init__(self, gb_model_path: str, tcn_model_path: str, device: str = 'auto'):
        """
        Initialize the prediction engine.
        
        Args:
            gb_model_path: Path to saved GradientBoosting model
            tcn_model_path: Path to saved TCN model
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        self.device = self._setup_device(device)
        self.gb_model = self._load_gb_model(gb_model_path)
        self.tcn_model, self.config = self._load_tcn_model(tcn_model_path)
        
        logger.info(f"Prediction engine initialized on {self.device}")
        logger.info(f"Model config: {self.config}")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device"""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.device(device)
    
    def _load_gb_model(self, gb_model_path: str):
        """Load GradientBoosting baseline model"""
        with open(gb_model_path, 'rb') as f:
            gb_model = pickle.load(f)
        logger.info(f"Loaded GradientBoosting model from {gb_model_path}")
        return gb_model
    
    def _load_tcn_model(self, tcn_model_path: str):
        """Load TCN model and configuration"""
        checkpoint = torch.load(tcn_model_path, map_location=self.device)
        config = checkpoint['config']
        
        # Recreate model architecture
        model = AutoregressiveTCNModel(
            static_features_size=config.get('static_features_size', 50),  # Will be updated
            seq_length=config['seq_length'],
            output_size=1,
            num_channels=config['num_channels'],
            kernel_size=config['kernel_size'],
            dropout=config['dropout'],
            num_layers=config.get('num_layers', 8)
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        logger.info(f"Loaded TCN model from {tcn_model_path}")
        return model, config
    
    def prepare_static_features(self, timestamp: pd.Timestamp, 
                              weather_data: Dict = None,
                              special_events: List[str] = None,
                              school_holiday: bool = False) -> np.ndarray:
        """
        Prepare static features for a given timestamp.
        This should match the feature engineering from training.
        
        Args:
            timestamp: Prediction timestamp
            weather_data: Optional weather information
            special_events: List of special events
            school_holiday: Whether it's a school holiday
        """
        # Basic temporal features
        features = {
            'hour': timestamp.hour,
            'day_of_week': timestamp.dayofweek,
            'month': timestamp.month,
            'is_weekend': int(timestamp.dayofweek >= 5),
            'is_holiday': int(school_holiday),
            'quarter': timestamp.quarter,
            'day_of_month': timestamp.day,
            'week_of_year': timestamp.isocalendar()[1],
        }
        
        # Cyclical encoding for temporal features
        features.update({
            'hour_sin': np.sin(2 * np.pi * timestamp.hour / 24),
            'hour_cos': np.cos(2 * np.pi * timestamp.hour / 24),
            'day_sin': np.sin(2 * np.pi * timestamp.dayofweek / 7),
            'day_cos': np.cos(2 * np.pi * timestamp.dayofweek / 7),
            'month_sin': np.sin(2 * np.pi * timestamp.month / 12),
            'month_cos': np.cos(2 * np.pi * timestamp.month / 12),
        })
        
        # Weather features (if available)
        if weather_data:
            features.update({
                'temperature': weather_data.get('temperature', 20.0),
                'humidity': weather_data.get('humidity', 50.0),
                'precipitation': weather_data.get('precipitation', 0.0),
                'wind_speed': weather_data.get('wind_speed', 5.0),
                'cloud_cover': weather_data.get('cloud_cover', 50.0),
            })
        else:
            # Default weather values
            features.update({
                'temperature': 20.0,
                'humidity': 50.0,
                'precipitation': 0.0,
                'wind_speed': 5.0,
                'cloud_cover': 50.0,
            })
        
        # Special events
        if special_events:
            features['has_special_event'] = 1
            features['num_special_events'] = len(special_events)
        else:
            features['has_special_event'] = 0
            features['num_special_events'] = 0
        
        # Convert to array (maintain consistent order)
        feature_names = sorted(features.keys())
        feature_array = np.array([features[name] for name in feature_names])
        
        return feature_array.astype(np.float32)
    
    def predict_sequence(self, start_timestamp: pd.Timestamp,
                        prediction_steps: int,
                        initial_sequence: Optional[List[Tuple[float, float]]] = None,
                        uncertainty_samples: int = 100,
                        weather_forecast: Optional[List[Dict]] = None,
                        events_schedule: Optional[Dict] = None) -> Dict:
        """
        Generate autoregressive predictions for multiple time steps.
        
        Args:
            start_timestamp: Starting timestamp for predictions
            prediction_steps: Number of future steps to predict
            initial_sequence: Initial sequence of (wait_time, residual) pairs
            uncertainty_samples: Number of Monte Carlo samples for uncertainty
            weather_forecast: List of weather forecasts for each step
            events_schedule: Dictionary of {timestamp: events} for special events
        
        Returns:
            Dictionary with predictions, confidence intervals, and metadata
        """
        logger.info(f"Generating {prediction_steps} predictions starting from {start_timestamp}")
        
        # Generate timestamps for prediction
        freq = '15T'  # 15-minute intervals (adjust based on your data)
        prediction_timestamps = pd.date_range(
            start=start_timestamp,
            periods=prediction_steps,
            freq=freq
        )
        
        # Initialize sequence
        if initial_sequence is None:
            # Use default values if no initial sequence provided
            initial_sequence = [(5.0, 0.0)] * self.config['seq_length']
        elif len(initial_sequence) < self.config['seq_length']:
            # Pad with zeros if sequence too short
            padding_needed = self.config['seq_length'] - len(initial_sequence)
            initial_sequence = [(0.0, 0.0)] * padding_needed + initial_sequence
        
        # Convert initial sequence to proper format
        current_sequence = []
        for wait_time, residual in initial_sequence[-self.config['seq_length']:]:
            current_sequence.extend([wait_time, residual])
        
        # Store predictions
        predictions = []
        gb_predictions = []
        residual_predictions = []
        uncertainty_bounds = []
        
        with torch.no_grad():
            for step, timestamp in enumerate(prediction_timestamps):
                # Get weather data for this step
                weather_data = None
                if weather_forecast and step < len(weather_forecast):
                    weather_data = weather_forecast[step]
                
                # Get special events for this step
                special_events = []
                if events_schedule and timestamp in events_schedule:
                    special_events = events_schedule[timestamp]
                
                # Prepare static features
                static_features = self.prepare_static_features(
                    timestamp, weather_data, special_events
                )
                
                # Generate Monte Carlo samples for uncertainty estimation
                step_predictions = []
                step_gb_predictions = []
                step_residual_predictions = []
                
                for sample in range(uncertainty_samples):
                    # Add noise to sequence for uncertainty estimation
                    if sample > 0:  # First sample uses clean sequence
                        noise_factor = 0.05  # Small noise for uncertainty
                        noisy_sequence = []
                        for i in range(0, len(current_sequence), 2):
                            wait_time = current_sequence[i]
                            residual = current_sequence[i + 1]
                            
                            wait_noise = np.random.normal(0, max(0.1, wait_time * noise_factor))
                            residual_noise = np.random.normal(0, max(0.05, abs(residual) * noise_factor))
                            
                            noisy_sequence.extend([
                                max(0, wait_time + wait_noise),
                                residual + residual_noise
                            ])
                        sequence_to_use = noisy_sequence
                    else:
                        sequence_to_use = current_sequence.copy()
                    
                    # Prepare model input
                    model_input = np.concatenate([static_features, sequence_to_use])
                    model_input_tensor = torch.FloatTensor(model_input).unsqueeze(0).to(self.device)
                    
                    # Get GB baseline prediction
                    gb_pred = self.gb_model.predict(static_features.reshape(1, -1))[0]
                    
                    # Get TCN residual prediction
                    residual_pred = self.tcn_model(model_input_tensor).cpu().numpy().flatten()[0]
                    
                    # Combined prediction
                    combined_pred = gb_pred + residual_pred
                    combined_pred = max(0, combined_pred)  # Ensure non-negative
                    
                    step_predictions.append(combined_pred)
                    step_gb_predictions.append(gb_pred)
                    step_residual_predictions.append(residual_pred)
                
                # Calculate statistics
                mean_pred = np.mean(step_predictions)
                std_pred = np.std(step_predictions)
                lower_bound = np.percentile(step_predictions, 5)   # 90% confidence interval
                upper_bound = np.percentile(step_predictions, 95)
                
                mean_gb = np.mean(step_gb_predictions)
                mean_residual = np.mean(step_residual_predictions)
                
                # Store results
                predictions.append(mean_pred)
                gb_predictions.append(mean_gb)
                residual_predictions.append(mean_residual)
                uncertainty_bounds.append((lower_bound, upper_bound, std_pred))
                
                # Update sequence for next prediction (use mean prediction)
                current_sequence = current_sequence[2:] + [mean_pred, mean_residual]
                
                # Log progress
                if (step + 1) % 50 == 0:
                    logger.info(f"Generated {step + 1}/{prediction_steps} predictions")
        
        # Prepare results
        results = {
            'timestamps': prediction_timestamps,
            'predictions': np.array(predictions),
            'gb_baseline': np.array(gb_predictions),
            'tcn_residuals': np.array(residual_predictions),
            'uncertainty_lower': np.array([ub[0] for ub in uncertainty_bounds]),
            'uncertainty_upper': np.array([ub[1] for ub in uncertainty_bounds]),
            'uncertainty_std': np.array([ub[2] for ub in uncertainty_bounds]),
            'config': self.config,
            'start_timestamp': start_timestamp,
            'prediction_steps': prediction_steps
        }
        
        logger.info(f"Completed {prediction_steps} predictions")
        logger.info(f"Mean prediction: {np.mean(predictions):.2f} ± {np.mean([ub[2] for ub in uncertainty_bounds]):.2f}")
        
        return results


class WaitTimePredictionVisualizer:
    """
    Comprehensive visualization suite for wait time predictions.
    """
    
    def __init__(self, predictions: Dict):
        """
        Initialize visualizer with prediction results.
        
        Args:
            predictions: Dictionary from WaitTimePredictionEngine.predict_sequence()
        """
        self.predictions = predictions
        self.timestamps = predictions['timestamps']
        self.pred_values = predictions['predictions']
        self.gb_baseline = predictions['gb_baseline']
        self.tcn_residuals = predictions['tcn_residuals']
        self.uncertainty_lower = predictions['uncertainty_lower']
        self.uncertainty_upper = predictions['uncertainty_upper']
        self.uncertainty_std = predictions['uncertainty_std']
        
    def create_time_series_plot(self, figsize: Tuple[int, int] = (15, 8),
                               save_path: Optional[str] = None) -> plt.Figure:
        """Create comprehensive time series plot with uncertainty bands"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, 
                                           height_ratios=[3, 2, 1], sharex=True)
        
        # Main prediction plot
        ax1.plot(self.timestamps, self.pred_values, 'b-', linewidth=2, 
                label='Combined Prediction', alpha=0.8)
        ax1.plot(self.timestamps, self.gb_baseline, 'g--', linewidth=1.5, 
                label='GB Baseline', alpha=0.7)
        
        # Uncertainty band
        ax1.fill_between(self.timestamps, self.uncertainty_lower, self.uncertainty_upper,
                        alpha=0.3, color='blue', label='90% Confidence Interval')
        
        ax1.set_ylabel('Wait Time (minutes)', fontsize=12)
        ax1.set_title('Wait Time Predictions with Uncertainty', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # TCN residuals plot
        ax2.plot(self.timestamps, self.tcn_residuals, 'r-', linewidth=1.5, 
                label='TCN Residuals', alpha=0.7)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_ylabel('Residual (minutes)', fontsize=12)
        ax2.set_title('TCN Model Residual Predictions', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Uncertainty plot
        ax3.plot(self.timestamps, self.uncertainty_std, 'orange', linewidth=1.5, 
                label='Prediction Std Dev', alpha=0.8)
        ax3.set_ylabel('Uncertainty', fontsize=12)
        ax3.set_xlabel('Time', fontsize=12)
        ax3.set_title('Prediction Uncertainty', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Format x-axis
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        ax3.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Time series plot saved to {save_path}")
        
        return fig
    
    def create_interactive_dashboard(self, save_path: Optional[str] = None) -> go.Figure:
        """Create interactive Plotly dashboard"""
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Wait Time Predictions', 'Model Components', 'Uncertainty Analysis'),
            vertical_spacing=0.08,
            row_heights=[0.5, 0.3, 0.2]
        )
        
        # Main prediction plot
        fig.add_trace(
            go.Scatter(
                x=self.timestamps, y=self.pred_values,
                mode='lines', name='Combined Prediction',
                line=dict(color='blue', width=3),
                hovertemplate='<b>Time:</b> %{x}<br><b>Wait Time:</b> %{y:.1f} min<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Uncertainty band
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([self.timestamps, self.timestamps[::-1]]),
                y=np.concatenate([self.uncertainty_upper, self.uncertainty_lower[::-1]]),
                fill='toself', fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='90% Confidence Interval',
                hoverinfo='skip'
            ),
            row=1, col=1
        )
        
        # GB baseline
        fig.add_trace(
            go.Scatter(
                x=self.timestamps, y=self.gb_baseline,
                mode='lines', name='GB Baseline',
                line=dict(color='green', width=2, dash='dash'),
                hovertemplate='<b>Time:</b> %{x}<br><b>Baseline:</b> %{y:.1f} min<extra></extra>'
            ),
            row=2, col=1
        )
        
        # TCN residuals
        fig.add_trace(
            go.Scatter(
                x=self.timestamps, y=self.tcn_residuals,
                mode='lines', name='TCN Residuals',
                line=dict(color='red', width=2),
                hovertemplate='<b>Time:</b> %{x}<br><b>Residual:</b> %{y:.1f} min<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Zero line for residuals
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=2, col=1)
        
        # Uncertainty
        fig.add_trace(
            go.Scatter(
                x=self.timestamps, y=self.uncertainty_std,
                mode='lines', name='Prediction Uncertainty',
                line=dict(color='orange', width=2),
                hovertemplate='<b>Time:</b> %{x}<br><b>Std Dev:</b> %{y:.2f} min<extra></extra>'
            ),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text='Interactive Wait Time Prediction Dashboard',
                x=0.5,
                font=dict(size=16)
            ),
            height=900,
            showlegend=True,
            hovermode='x unified'
        )
        
        # Update y-axes labels
        fig.update_yaxes(title_text="Wait Time (min)", row=1, col=1)
        fig.update_yaxes(title_text="Component Value", row=2, col=1)
        fig.update_yaxes(title_text="Uncertainty", row=3, col=1)
        fig.update_xaxes(title_text="Time", row=3, col=1)
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Interactive dashboard saved to {save_path}")
        
        return fig
    
    def create_distribution_analysis(self, figsize: Tuple[int, int] = (12, 8),
                                   save_path: Optional[str] = None) -> plt.Figure:
        """Analyze prediction distributions and patterns"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Prediction distribution
        ax1.hist(self.pred_values, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(np.mean(self.pred_values), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(self.pred_values):.1f}')
        ax1.set_xlabel('Wait Time (minutes)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Predicted Wait Times')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Hourly patterns
        hours = self.timestamps.hour
        hourly_mean = [self.pred_values[hours == h].mean() for h in range(24)]
        ax2.bar(range(24), hourly_mean, alpha=0.7, color='green')
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Average Wait Time (minutes)')
        ax2.set_title('Hourly Wait Time Patterns')
        ax2.set_xticks(range(0, 24, 3))
        ax2.grid(True, alpha=0.3)
        
        # Model components comparison
        ax3.scatter(self.gb_baseline, self.tcn_residuals, alpha=0.6, s=20)
        ax3.set_xlabel('GB Baseline (minutes)')
        ax3.set_ylabel('TCN Residuals (minutes)')
        ax3.set_title('GB Baseline vs TCN Residuals')
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax3.grid(True, alpha=0.3)
        
        # Uncertainty over time
        uncertainty_trend = pd.Series(self.uncertainty_std, index=self.timestamps).rolling('6H').mean()
        ax4.plot(self.timestamps, uncertainty_trend, color='orange', linewidth=2)
        ax4.set_xlabel('Time')
        ax4.set_ylabel('6-Hour Rolling Avg Uncertainty')
        ax4.set_title('Uncertainty Trends')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Distribution analysis saved to {save_path}")
        
        return fig
    
    def create_summary_stats(self) -> Dict:
        """Generate comprehensive summary statistics"""
        stats = {
            'prediction_summary': {
                'mean': float(np.mean(self.pred_values)),
                'std': float(np.std(self.pred_values)),
                'min': float(np.min(self.pred_values)),
                'max': float(np.max(self.pred_values)),
                'median': float(np.median(self.pred_values)),
                'q25': float(np.percentile(self.pred_values, 25)),
                'q75': float(np.percentile(self.pred_values, 75))
            },
            'uncertainty_summary': {
                'mean_uncertainty': float(np.mean(self.uncertainty_std)),
                'max_uncertainty': float(np.max(self.uncertainty_std)),
                'min_uncertainty': float(np.min(self.uncertainty_std)),
                'uncertainty_trend': 'increasing' if self.uncertainty_std[-1] > self.uncertainty_std[0] else 'decreasing'
            },
            'model_components': {
                'gb_contribution_mean': float(np.mean(self.gb_baseline)),
                'tcn_contribution_mean': float(np.mean(self.tcn_residuals)),
                'gb_vs_tcn_correlation': float(np.corrcoef(self.gb_baseline, self.tcn_residuals)[0, 1])
            },
            'temporal_patterns': {
                'peak_hour': int(self.timestamps[np.argmax(self.pred_values)].hour),
                'lowest_hour': int(self.timestamps[np.argmin(self.pred_values)].hour),
                'peak_prediction': float(np.max(self.pred_values)),
                'lowest_prediction': float(np.min(self.pred_values))
            }
        }
        
        return stats


def main():
    """Main inference script"""
    parser = argparse.ArgumentParser(description="Predict wait times using trained models")
    parser.add_argument('--ride', required=True, help='Ride name')
    parser.add_argument('--models-dir', default='models/cached_scheduled_sampling',
                       help='Directory containing trained models')
    parser.add_argument('--start-time', 
                       help='Start time for prediction (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--prediction-days', type=float, default=1.0,
                       help='Number of days to predict into the future')
    parser.add_argument('--output-dir', default='predictions',
                       help='Output directory for results and visualizations')
    parser.add_argument('--uncertainty-samples', type=int, default=100,
                       help='Number of Monte Carlo samples for uncertainty estimation')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto',
                       help='Computation device')
    
    args = parser.parse_args()
    
    # Setup paths
    ride_normalized = args.ride.replace(' ', '_')
    gb_model_path = os.path.join(args.models_dir, f"{ride_normalized}_gb_baseline.pkl")
    tcn_model_path = os.path.join(args.models_dir, f"{ride_normalized}_cached_scheduled_sampling_tcn.pt")
    
    # Validate model files exist
    if not os.path.exists(gb_model_path):
        raise FileNotFoundError(f"GradientBoosting model not found: {gb_model_path}")
    if not os.path.exists(tcn_model_path):
        raise FileNotFoundError(f"TCN model not found: {tcn_model_path}")
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse start time
    if args.start_time:
        start_timestamp = pd.to_datetime(args.start_time)
    else:
        start_timestamp = pd.Timestamp.now().floor('15T')  # Round to nearest 15 minutes
    
    # Calculate prediction steps (assuming 15-minute intervals)
    prediction_steps = int(args.prediction_days * 24 * 4)  # 4 steps per hour
    
    logger.info(f"Starting wait time prediction for {args.ride}")
    logger.info(f"Prediction period: {start_timestamp} to {start_timestamp + timedelta(days=args.prediction_days)}")
    logger.info(f"Total prediction steps: {prediction_steps}")
    
    # Initialize prediction engine
    engine = WaitTimePredictionEngine(gb_model_path, tcn_model_path, args.device)
    
    # Generate predictions
    logger.info("Generating predictions...")
    predictions = engine.predict_sequence(
        start_timestamp=start_timestamp,
        prediction_steps=prediction_steps,
        uncertainty_samples=args.uncertainty_samples
    )
    
    # Create visualizations
    logger.info("Creating visualizations...")
    visualizer = WaitTimePredictionVisualizer(predictions)
    
    # Generate all plots
    timestamp_str = start_timestamp.strftime('%Y%m%d_%H%M')
    
    # Time series plot
    ts_fig = visualizer.create_time_series_plot()
    ts_path = os.path.join(args.output_dir, f"{ride_normalized}_predictions_{timestamp_str}.png")
    ts_fig.savefig(ts_path, dpi=300, bbox_inches='tight')
    plt.close(ts_fig)
    
    # Interactive dashboard
    dashboard_path = os.path.join(args.output_dir, f"{ride_normalized}_dashboard_{timestamp_str}.html")
    interactive_fig = visualizer.create_interactive_dashboard(dashboard_path)
    
    # Distribution analysis
    dist_fig = visualizer.create_distribution_analysis()
    dist_path = os.path.join(args.output_dir, f"{ride_normalized}_analysis_{timestamp_str}.png")
    dist_fig.savefig(dist_path, dpi=300, bbox_inches='tight')
    plt.close(dist_fig)
    
    # Generate summary statistics
    summary_stats = visualizer.create_summary_stats()
    
    # Save predictions to CSV
    results_df = pd.DataFrame({
        'timestamp': predictions['timestamps'],
        'predicted_wait_time': predictions['predictions'],
        'gb_baseline': predictions['gb_baseline'],
        'tcn_residual': predictions['tcn_residuals'],
        'uncertainty_lower': predictions['uncertainty_lower'],
        'uncertainty_upper': predictions['uncertainty_upper'],
        'uncertainty_std': predictions['uncertainty_std']
    })
    
    csv_path = os.path.join(args.output_dir, f"{ride_normalized}_predictions_{timestamp_str}.csv")
    results_df.to_csv(csv_path, index=False)
    
    # Save summary statistics to JSON
    stats_path = os.path.join(args.output_dir, f"{ride_normalized}_summary_{timestamp_str}.json")
    with open(stats_path, 'w') as f:
        json.dump(summary_stats, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "="*60)
    print(f"WAIT TIME PREDICTION SUMMARY - {args.ride}")
    print("="*60)
    print(f"Prediction Period: {start_timestamp} to {start_timestamp + timedelta(days=args.prediction_days)}")
    print(f"Total Predictions: {prediction_steps}")
    print(f"\nPrediction Statistics:")
    print(f"  Average Wait Time: {summary_stats['prediction_summary']['mean']:.1f} ± {summary_stats['uncertainty_summary']['mean_uncertainty']:.1f} minutes")
    print(f"  Range: {summary_stats['prediction_summary']['min']:.1f} - {summary_stats['prediction_summary']['max']:.1f} minutes")
    print(f"  Peak Hour: {summary_stats['temporal_patterns']['peak_hour']:02d}:00 ({summary_stats['temporal_patterns']['peak_prediction']:.1f} min)")
    print(f"  Lowest Hour: {summary_stats['temporal_patterns']['lowest_hour']:02d}:00 ({summary_stats['temporal_patterns']['lowest_prediction']:.1f} min)")
    print(f"\nModel Components:")
    print(f"  GB Baseline Contribution: {summary_stats['model_components']['gb_contribution_mean']:.1f} minutes")
    print(f"  TCN Residual Contribution: {summary_stats['model_components']['tcn_contribution_mean']:.1f} minutes")
    print(f"\nUncertainty Analysis:")
    print(f"  Average Uncertainty: {summary_stats['uncertainty_summary']['mean_uncertainty']:.2f} minutes")
    print(f"  Max Uncertainty: {summary_stats['uncertainty_summary']['max_uncertainty']:.2f} minutes")
    print(f"  Uncertainty Trend: {summary_stats['uncertainty_summary']['uncertainty_trend']}")
    print(f"\nOutput Files:")
    print(f"  Predictions CSV: {csv_path}")
    print(f"  Time Series Plot: {ts_path}")
    print(f"  Analysis Plot: {dist_path}")
    print(f"  Interactive Dashboard: {dashboard_path}")
    print(f"  Summary Statistics: {stats_path}")
    print("="*60)


if __name__ == "__main__":
    main()