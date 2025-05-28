import os
import pickle
import logging
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from TCN.models.tcn_model import AutoregressiveTCNModel
from TCN.datasets.data_utils import preprocess_data, create_features

logger = logging.getLogger(__name__)


class WaitTimePredictor:
    """
    Inference class for wait time prediction using trained TCN + GradientBoosting models.
    """
    
    def __init__(self, ride_name: str, model_dir: str = "models/cached_scheduled_sampling", 
                 device: Optional[str] = None):
        """
        Initialize predictor for a specific ride.
        
        Args:
            ride_name: Name of the ride
            model_dir: Directory containing saved models
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.ride_name = ride_name
        self.model_dir = Path(model_dir)
        self.device = self._setup_device(device)
        
        # Load models
        self.gb_model, self.tcn_model, self.config = self._load_models()
        
        # Extract model configuration
        self.static_features_size = self.config['static_features_size']
        self.seq_length = self.config['seq_length']
        self.static_feature_cols = None  # Will be set during preprocessing
        
        logger.info(f"Initialized predictor for ride '{ride_name}' on {self.device}")
    
    def _setup_device(self, device: Optional[str]) -> torch.device:
        """Setup compute device"""
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def _load_models(self) -> Tuple[object, torch.nn.Module, dict]:
        """Load trained models and configuration"""
        ride_name_normalized = self.ride_name.replace(' ', '_')
        
        # Load GradientBoosting model
        gb_path = self.model_dir / f"{ride_name_normalized}_gb_baseline.pkl"
        with open(gb_path, "rb") as f:
            gb_model = pickle.load(f)
        
        # Load TCN model and config
        tcn_path = self.model_dir / f"{ride_name_normalized}_cached_scheduled_sampling_tcn.pt"
        checkpoint = torch.load(tcn_path, map_location=self.device)
        
        config = checkpoint['config']
        
        # Debug: Print what we're loading
        logger.info(f"Loading model configuration:")
        logger.info(f"  Config keys: {list(config.keys())}")
        logger.info(f"  Original config: {config}")
        
        # The key issue: we need to calculate the correct tcn_input_size
        # From your tcn_model.py, the TCN input size is: static_features_size + 2 (for wait_time + residual per timestep)
        
        # First, get the static_features_size
        if 'static_features_size' in config:
            static_features_size = config['static_features_size']
            logger.info(f"  Using saved static_features_size: {static_features_size}")
        else:
            # Try to infer from saved static_feature_cols
            if 'static_feature_cols' in config and config['static_feature_cols']:
                static_features_size = len(config['static_feature_cols'])
                config['static_features_size'] = static_features_size
                logger.info(f"  Calculated static_features_size from static_feature_cols: {static_features_size}")
            else:
                # Infer from the actual model weights
                state_dict = checkpoint['model_state_dict']
                
                # Handle torch.compile prefixes first
                if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
                    temp_state_dict = {}
                    for key, value in state_dict.items():
                        if key.startswith('_orig_mod.'):
                            new_key = key[10:]
                            temp_state_dict[new_key] = value
                        else:
                            temp_state_dict[key] = value
                    state_dict = temp_state_dict
                
                # Look for the input layer to infer the total input size
                tcn_input_size = None
                for key, tensor in state_dict.items():
                    if 'tcn.network.0.conv1' in key and 'weight' in key and 'original1' in key:
                        # This is the first conv layer weight: [out_channels, in_channels, kernel_size]
                        tcn_input_size = tensor.shape[1]  # in_channels
                        logger.info(f"  Found TCN input size from {key}: {tcn_input_size}")
                        break
                    elif 'tcn.network.0.downsample.weight' in key:
                        # Downsample layer: [out_channels, in_channels, 1]
                        tcn_input_size = tensor.shape[1]
                        logger.info(f"  Found TCN input size from {key}: {tcn_input_size}")
                        break
                
                if tcn_input_size is None:
                    raise ValueError("Could not determine TCN input size from model weights")
                
                # Calculate static_features_size
                # From your model: tcn_input_size = static_features_size + 2
                static_features_size = tcn_input_size - 2
                config['static_features_size'] = static_features_size
                logger.info(f"  Calculated static_features_size: {static_features_size} (tcn_input_size={tcn_input_size} - 2)")
        
        # Ensure all required config parameters are present
        required_params = ['seq_length', 'num_channels', 'kernel_size', 'num_layers']
        for param in required_params:
            if param not in config:
                logger.error(f"Missing required parameter: {param}")
                raise ValueError(f"Configuration missing required parameter: {param}")
        
        # Set default values for optional parameters
        config.setdefault('dropout', 0.2)
        config.setdefault('output_size', 1)
        
        # Debug: Print final configuration
        logger.info(f"Final model configuration:")
        logger.info(f"  static_features_size: {config['static_features_size']}")
        logger.info(f"  seq_length: {config['seq_length']}")
        logger.info(f"  num_channels: {config['num_channels']}")
        logger.info(f"  kernel_size: {config['kernel_size']}")
        logger.info(f"  num_layers: {config['num_layers']}")
        logger.info(f"  dropout: {config['dropout']}")
        logger.info(f"  output_size: {config['output_size']}")
        
        # Create TCN model with correct configuration
        tcn_model = AutoregressiveTCNModel.from_config(config)
        logger.info(f"Created TCN model with tcn_input_size: {tcn_model.tcn_input_size}")
        
        # Handle torch.compile model states
        state_dict = checkpoint['model_state_dict']
        
        # Check if the state dict has torch.compile prefixes
        if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
            logger.info("Detected torch.compile model, removing _orig_mod. prefixes")
            # Remove the _orig_mod. prefix from all keys
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('_orig_mod.'):
                    new_key = key[10:]  # Remove '_orig_mod.' prefix (10 characters)
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict
        
        # Load the cleaned state dict
        try:
            tcn_model.load_state_dict(state_dict)
            logger.info("Successfully loaded model state dict")
        except RuntimeError as e:
            logger.error(f"Failed to load state dict: {e}")
            logger.error("Model configuration may not match saved weights")
            
            # Debug: show shape mismatches
            model_state = tcn_model.state_dict()
            for key in state_dict.keys():
                if key in model_state:
                    if state_dict[key].shape != model_state[key].shape:
                        logger.error(f"Shape mismatch for {key}: saved={state_dict[key].shape}, model={model_state[key].shape}")
                else:
                    logger.error(f"Key {key} not found in model")
            raise
            
        tcn_model.to(self.device)
        tcn_model.eval()
        
        return gb_model, tcn_model, config
    
    def preprocess_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess input data to match training data format exactly"""
        
        # Step 1: Apply same preprocessing as training (from data_utils.py)
        df = preprocess_data(df, self.ride_name)
        
        # Step 2: Create features exactly like training (from data_utils.py)
        # Static features are everything except wait_time and timestamp
        static_feature_cols = [col for col in df.columns 
                            if col not in ['wait_time', 'timestamp']]
        
        logger.info(f"Initial features after preprocessing: {len(static_feature_cols)}")
        
        # Step 3: Get expected training features from config if available
        expected_training_features = [
            'closed', 'is_german_holiday', 'is_swiss_holiday', 'is_french_holiday', 
            'day_of_week', 'temperature', 'rain', 'weekday', 'is_weekend', 
            'month_sin', 'month_cos', 'hour_sin', 'hour_cos', 'weekday_sin', 
            'weekday_cos', 'minute_sin', 'minute_cos', 'part_of_day_afternoon', 
            'part_of_day_evening', 'part_of_day_morning', 'part_of_day_night', 
            'season_fall', 'season_spring', 'season_summer', 'season_winter', 
            'year_2017', 'year_2018', 'year_2019', 'year_2020', 'year_2021', 
            'year_2022', 'year_2023', 'year_2024'
        ]
        
        # Try to use saved feature columns from config first
        if 'static_feature_cols' in self.config and self.config['static_feature_cols']:
            expected_training_features = self.config['static_feature_cols']
            logger.info(f"Using saved feature columns from model config: {len(expected_training_features)} features")
        else:
            logger.info(f"Using hardcoded expected features: {len(expected_training_features)} features")
        
        # Step 4: Filter to only include expected training features in exact order
        final_feature_cols = []
        missing_features = []
        
        for feature in expected_training_features:
            if feature in static_feature_cols:
                # Verify the feature is numeric
                if feature in df.columns:
                    if pd.api.types.is_numeric_dtype(df[feature]):
                        final_feature_cols.append(feature)
                    else:
                        logger.error(f"Feature '{feature}' exists but is not numeric (dtype: {df[feature].dtype})")
                        logger.error(f"Sample values: {df[feature].dropna().head(3).tolist()}")
                        
                        # Try to convert to numeric if it's a simple case
                        if feature == 'day_of_week' and df[feature].dtype == 'object':
                            logger.warning(f"Attempting to fix day_of_week encoding...")
                            # If it's day names, convert back to numbers
                            day_mapping = {
                                'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
                                'Friday': 4, 'Saturday': 5, 'Sunday': 6
                            }
                            if all(val in day_mapping for val in df[feature].dropna().unique()):
                                df[feature] = df[feature].map(day_mapping)
                                final_feature_cols.append(feature)
                                logger.info(f"Successfully converted day_of_week to numeric")
                            else:
                                missing_features.append(feature)
                        else:
                            missing_features.append(feature)
                else:
                    missing_features.append(feature)
            else:
                missing_features.append(feature)
        
        # Report findings
        extra_features = [f for f in static_feature_cols if f not in expected_training_features]
        
        logger.info(f"Feature matching results:")
        logger.info(f"  Expected features: {len(expected_training_features)}")
        logger.info(f"  Found and valid: {len(final_feature_cols)}")
        logger.info(f"  Missing features: {missing_features}")
        logger.info(f"  Extra features (ignored): {extra_features}")
        
        # Verify we have the exact number expected
        expected_features = self.config['static_features_size']
        
        if len(final_feature_cols) == expected_features:
            logger.info(f"✅ Feature count matches exactly: {len(final_feature_cols)}")
            self.static_feature_cols = final_feature_cols
        elif len(final_feature_cols) < expected_features:
            logger.error(f"❌ Not enough features: {len(final_feature_cols)} < {expected_features}")
            logger.error(f"Missing features: {missing_features}")
            logger.error(f"This will cause model input size mismatch!")
            
            # Show detailed debugging info
            logger.error("Available columns in dataframe:")
            for col in df.columns:
                logger.error(f"  {col}: {df[col].dtype}")
            
            raise ValueError(f"Feature count mismatch: need {expected_features} features, got {len(final_feature_cols)}")
        else:
            logger.warning(f"⚠️ Too many features: {len(final_feature_cols)} > {expected_features}")
            logger.warning(f"This should not happen with the fixed preprocessing!")
            # Take exactly the expected number in the correct order
            self.static_feature_cols = final_feature_cols[:expected_features]
        
        logger.info(f"Final selected features ({len(self.static_feature_cols)}):")
        for i, feature in enumerate(self.static_feature_cols):
            logger.info(f"  {i+1:2d}. {feature}")
        
        return df
    
    def predict_single(self, static_features: np.ndarray, 
                      historical_sequence: List[Tuple[float, float]]) -> Dict[str, float]:
        """
        Predict wait time for a single sample.
        
        Args:
            static_features: Array of static features
            historical_sequence: List of (wait_time, residual) tuples for past timesteps
            
        Returns:
            Dictionary with predictions and components
        """
        # Ensure we have the right sequence length
        if len(historical_sequence) < self.seq_length:
            # Pad with zeros at the beginning
            padding_needed = self.seq_length - len(historical_sequence)
            historical_sequence = [(0.0, 0.0)] * padding_needed + historical_sequence
        elif len(historical_sequence) > self.seq_length:
            # Take the most recent timesteps
            historical_sequence = historical_sequence[-self.seq_length:]
        
        # Flatten the historical sequence
        autoregressive_features = []
        for wait_time, residual in historical_sequence:
            autoregressive_features.extend([wait_time, residual])
        
        # Combine features
        combined_features = np.concatenate([
            static_features,
            np.array(autoregressive_features, dtype=np.float32)
        ])
        
        # Get baseline prediction
        gb_pred = self.gb_model.predict(static_features.reshape(1, -1))[0]
        
        # Get TCN residual prediction
        with torch.no_grad():
            tcn_input = torch.FloatTensor(combined_features).unsqueeze(0).to(self.device)
            residual_pred = self.tcn_model(tcn_input).cpu().numpy().flatten()[0]
        
        # Combined prediction
        combined_pred = gb_pred + residual_pred
        
        return {
            'wait_time_prediction': max(0, combined_pred),
            'baseline_prediction': gb_pred,
            'residual_prediction': residual_pred
        }
    
    def predict_batch(self, df: pd.DataFrame, use_autoregressive: bool = True) -> pd.DataFrame:
        """
        Predict wait times with FIXED same-day autoregressive logic.
        
        Key improvement: Only uses predictions within the same operational day,
        preventing error accumulation across days.
        
        Args:
            df: DataFrame with required features and historical data
            use_autoregressive: If True, use model predictions autoregressively within same day;
                              if False, use ground truth (teacher forcing)
        
        Returns:
            DataFrame with predictions added
        """
        # Preprocess if needed
        if self.static_feature_cols is None:
            df = self.preprocess_input(df)
        
        # Ensure timestamps are datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Get static features
        X_static = df[self.static_feature_cols].values
        
        # Get baseline predictions
        gb_predictions = self.gb_model.predict(X_static)
        
        # Initialize results
        predictions = []
        residuals = []
        
        # For autoregressive prediction, track predictions by date AND index
        if use_autoregressive:
            prediction_history = {}  # Track predictions for each timestamp
            daily_prediction_starts = {}  # Track when we start predicting each day
            logger.info(f"🔧 Starting FIXED autoregressive prediction (same-day logic)")
            logger.info(f"   Sequence length: {self.seq_length} timesteps")
            logger.info(f"   Opening hour: {self.config.get('opening_hour', 9)}")
        
        for idx in range(len(df)):
            if idx < self.seq_length:
                # Not enough history, use baseline only
                predictions.append(gb_predictions[idx])
                residuals.append(0.0)
                continue
            
            current_timestamp = df.iloc[idx]['timestamp']
            current_date = current_timestamp.date()
            current_hour = current_timestamp.hour
            
            # Track when we start making predictions for each day (at opening time)
            if use_autoregressive and current_date not in daily_prediction_starts:
                if current_hour >= self.config.get('opening_hour', 9):  # Opening hour (default 9 AM)
                    daily_prediction_starts[current_date] = idx
                    if len(daily_prediction_starts) <= 5:  # Log first few days
                        logger.info(f"   📅 Starting predictions for {current_date} at index {idx}")
            
            # Build historical sequence with SAME-DAY logic
            historical_sequence = []
            same_day_predictions_used = 0
            cross_day_ground_truth_used = 0
            
            for hist_idx in range(idx - self.seq_length, idx):
                hist_timestamp = df.iloc[hist_idx]['timestamp']
                hist_date = hist_timestamp.date()
                hist_hour = hist_timestamp.hour
                
                # Determine whether to use predictions or ground truth
                use_prediction = False
                
                if use_autoregressive and hist_idx in prediction_history:
                    # We have a prediction for this timestep
                    if hist_date == current_date:
                        # Same day: use prediction only if it's during operational hours
                        # and we've started making predictions for this day
                        if (current_date in daily_prediction_starts and 
                            hist_idx >= daily_prediction_starts[current_date] and
                            hist_hour >= self.config.get('opening_hour', 9)):
                            use_prediction = True
                            same_day_predictions_used += 1
                        # Otherwise, use ground truth even for same day if before operational hours
                    else:
                        # Different day: always use ground truth to prevent error accumulation
                        use_prediction = False
                        cross_day_ground_truth_used += 1
                
                if use_prediction:
                    # Use previous prediction
                    wait_time = prediction_history[hist_idx]['wait_time']
                    residual = prediction_history[hist_idx]['residual']
                else:
                    # Use ground truth
                    wait_time = df.iloc[hist_idx]['wait_time']
                    residual = wait_time - gb_predictions[hist_idx]
                    cross_day_ground_truth_used += 1
                
                historical_sequence.append((wait_time, residual))
            
            # Log sampling statistics periodically
            if use_autoregressive and idx % 2000 == 0:
                total_sequence_length = len(historical_sequence)
                same_day_pct = (same_day_predictions_used / total_sequence_length) * 100
                ground_truth_pct = (cross_day_ground_truth_used / total_sequence_length) * 100
                
                logger.info(f"   📊 Index {idx} ({current_date} {current_hour:02d}:00): "
                           f"Same-day predictions: {same_day_pct:.1f}%, "
                           f"Ground truth: {ground_truth_pct:.1f}%")
            
            # Get prediction for current timestep
            static_features = X_static[idx]
            result = self.predict_single(static_features, historical_sequence)
            
            predictions.append(result['wait_time_prediction'])
            residuals.append(result['residual_prediction'])
            
            # Store for autoregressive use (only during operational hours)
            if use_autoregressive and current_hour >= self.config.get('opening_hour', 9):
                prediction_history[idx] = {
                    'wait_time': result['wait_time_prediction'],
                    'residual': result['residual_prediction']
                }
        
        # Add predictions to dataframe
        df['baseline_prediction'] = gb_predictions
        df['residual_prediction'] = residuals
        df['wait_time_prediction'] = predictions
        
        # Log final statistics
        if use_autoregressive:
            logger.info(f"✅ Completed FIXED autoregressive prediction:")
            logger.info(f"   📈 Total predictions: {len(predictions)}")
            logger.info(f"   📅 Days with predictions: {len(daily_prediction_starts)}")
            logger.info(f"   🎯 Should see MUCH better performance due to same-day logic!")
        
        return df
    
    def predict_sequence(self, initial_static_features: np.ndarray,
                        initial_history: List[Tuple[float, float]],
                        future_static_features: np.ndarray,
                        horizon: int) -> List[Dict[str, float]]:
        """
        Predict a sequence of future wait times autoregressively.
        
        Args:
            initial_static_features: Static features for the starting point
            initial_history: Historical sequence up to the starting point
            future_static_features: Static features for each future timestep (shape: [horizon, n_features])
            horizon: Number of future timesteps to predict
            
        Returns:
            List of prediction dictionaries for each future timestep
        """
        predictions = []
        history = initial_history.copy()
        
        for t in range(horizon):
            # Get static features for this timestep
            static_features = future_static_features[t] if t < len(future_static_features) else initial_static_features
            
            # Make prediction
            result = self.predict_single(static_features, history)
            predictions.append(result)
            
            # Update history for next prediction
            # Remove oldest entry and add new prediction
            history = history[1:] + [(result['wait_time_prediction'], result['residual_prediction'])]
        
        return predictions
    
    def evaluate_predictions(self, df: pd.DataFrame, target_col: str = 'wait_time') -> Dict[str, float]:
        """
        Evaluate predictions against ground truth.
        
        Args:
            df: DataFrame with predictions and ground truth
            target_col: Name of the target column
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Filter valid predictions (after seq_length)
        valid_df = df.iloc[self.seq_length:].copy()
        
        # Filter out closed rides if column exists
        if 'closed' in valid_df.columns:
            valid_df = valid_df[valid_df['closed'] == 0]
        
        y_true = valid_df[target_col].values
        y_pred = valid_df['wait_time_prediction'].values
        
        # Calculate metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # Calculate sMAPE
        non_zero_mask = y_true > 0.1
        if np.sum(non_zero_mask) > 0:
            y_true_nz = y_true[non_zero_mask]
            y_pred_nz = y_pred[non_zero_mask]
            smape = np.mean(np.abs(y_true_nz - y_pred_nz) / (np.abs(y_true_nz) + np.abs(y_pred_nz))) * 100
        else:
            smape = 0.0
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'smape': smape,
            'n_samples': len(y_true)
        }


def main():
    """Example usage of the predictor"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Wait time prediction inference")
    parser.add_argument('--ride', required=True, help='Ride name')
    parser.add_argument('--data', required=True, help='Path to input data (parquet file)')
    parser.add_argument('--output', help='Path to save predictions')
    parser.add_argument('--model-dir', default='models/cached_scheduled_sampling', 
                       help='Directory containing trained models')
    parser.add_argument('--autoregressive', action='store_true',
                       help='Use autoregressive prediction (default: teacher forcing)')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate predictions against ground truth')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize predictor
    predictor = WaitTimePredictor(args.ride, args.model_dir)
    
    # Load data
    df = pd.read_parquet(args.data)
    
    # Make predictions
    logger.info(f"Making predictions (autoregressive={args.autoregressive})...")
    predictions_df = predictor.predict_batch(df, use_autoregressive=args.autoregressive)
    
    # Evaluate if requested
    if args.evaluate and 'wait_time' in df.columns:
        metrics = predictor.evaluate_predictions(predictions_df)
        logger.info(f"Evaluation metrics: {metrics}")
    
    # Save predictions if output path provided
    if args.output:
        predictions_df.to_parquet(args.output)
        logger.info(f"Predictions saved to {args.output}")
    
    # Example of sequence prediction
    if len(df) >= predictor.seq_length:
        logger.info("\nExample sequence prediction:")
        
        # Get initial state from middle of dataset
        start_idx = len(df) // 2
        # Note: Fixed reference to static features
        X_static = df[predictor.static_feature_cols].values
        initial_static = X_static[start_idx]
        
        # Build initial history
        initial_history = []
        for i in range(start_idx - predictor.seq_length, start_idx):
            wait_time = df.iloc[i]['wait_time']
            residual = wait_time - predictor.gb_model.predict(X_static[i].reshape(1, -1))[0]
            initial_history.append((wait_time, residual))
        
        # Predict next 12 timesteps (3 hours with 15-min intervals)
        horizon = 12
        future_static = X_static[start_idx:start_idx+horizon]
        
        sequence_predictions = predictor.predict_sequence(
            initial_static, initial_history, future_static, horizon
        )
        
        for t, pred in enumerate(sequence_predictions):
            logger.info(f"  t+{t+1}: {pred['wait_time_prediction']:.1f} min")


if __name__ == "__main__":
    main()