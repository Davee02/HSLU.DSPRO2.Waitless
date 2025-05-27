import os
import pickle
import logging
import numpy as np
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
        config['static_features_size'] = len(config.get('static_feature_cols', []))
        
        # Create and load TCN model
        tcn_model = AutoregressiveTCNModel.from_config(config)
        tcn_model.load_state_dict(checkpoint['model_state_dict'])
        tcn_model.to(self.device)
        tcn_model.eval()
        
        return gb_model, tcn_model, config
    
    def preprocess_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess input data"""
        # Apply same preprocessing as training
        df = preprocess_data(df, self.ride_name)
        df, self.static_feature_cols = create_features(df)
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
        Predict wait times for a batch of samples.
        
        Args:
            df: DataFrame with required features and historical data
            use_autoregressive: If True, use model predictions autoregressively;
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
        
        # For autoregressive prediction, we need to track predictions
        if use_autoregressive:
            prediction_history = {}  # Track predictions for each timestamp
        
        for idx in range(len(df)):
            if idx < self.seq_length:
                # Not enough history, use baseline only
                predictions.append(gb_predictions[idx])
                residuals.append(0.0)
                continue
            
            # Build historical sequence
            historical_sequence = []
            
            for hist_idx in range(idx - self.seq_length, idx):
                if use_autoregressive and hist_idx in prediction_history:
                    # Use previous predictions
                    wait_time = prediction_history[hist_idx]['wait_time']
                    residual = prediction_history[hist_idx]['residual']
                else:
                    # Use ground truth
                    wait_time = df.iloc[hist_idx]['wait_time']
                    residual = wait_time - gb_predictions[hist_idx]
                
                historical_sequence.append((wait_time, residual))
            
            # Get prediction for current timestep
            static_features = X_static[idx]
            result = self.predict_single(static_features, historical_sequence)
            
            predictions.append(result['wait_time_prediction'])
            residuals.append(result['residual_prediction'])
            
            # Store for autoregressive use
            if use_autoregressive:
                prediction_history[idx] = {
                    'wait_time': result['wait_time_prediction'],
                    'residual': result['residual_prediction']
                }
        
        # Add predictions to dataframe
        df['baseline_prediction'] = gb_predictions
        df['residual_prediction'] = residuals
        df['wait_time_prediction'] = predictions
        
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
        initial_static = predictor.X_static[start_idx]
        
        # Build initial history
        initial_history = []
        for i in range(start_idx - predictor.seq_length, start_idx):
            wait_time = df.iloc[i]['wait_time']
            residual = wait_time - predictor.gb_model.predict(predictor.X_static[i].reshape(1, -1))[0]
            initial_history.append((wait_time, residual))
        
        # Predict next 12 timesteps (3 hours with 15-min intervals)
        horizon = 12
        future_static = predictor.X_static[start_idx:start_idx+horizon]
        
        sequence_predictions = predictor.predict_sequence(
            initial_static, initial_history, future_static, horizon
        )
        
        for t, pred in enumerate(sequence_predictions):
            logger.info(f"  t+{t+1}: {pred['wait_time_prediction']:.1f} min")


if __name__ == "__main__":
    main()