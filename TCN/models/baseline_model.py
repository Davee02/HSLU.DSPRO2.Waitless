import logging
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class GradientBoostingBaseline:
    """
    Wrapper for GradientBoosting baseline model with additional functionality.
    """
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the baseline model.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config or {}
        
        # Extract parameters from config
        self.n_estimators = self.config.get('gb_n_estimators', 100)
        self.learning_rate = self.config.get('gb_learning_rate', 0.1)
        self.max_depth = self.config.get('gb_max_depth', 6)
        self.min_samples_split = self.config.get('gb_min_samples_split', 10)
        self.min_samples_leaf = self.config.get('gb_min_samples_leaf', 5)
        self.random_state = self.config.get('random_state', 42)
        
        # Initialize the model
        self.model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state
        )
        
        self.is_fitted = False
        
    def fit(self, X, y):
        """Train the baseline model"""
        logger.info(f"Training GradientBoosting baseline with {len(X)} samples...")
        self.model.fit(X, y)
        self.is_fitted = True
        logger.info("Baseline model training completed")
        return self
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)
    
    def get_residuals(self, X, y):
        """Calculate residuals for TCN training"""
        predictions = self.predict(X)
        residuals = y - predictions
        return residuals, predictions
    
    def evaluate(self, X, y, exclude_closed=True, closed_mask=None):
        """
        Evaluate the baseline model.
        
        Args:
            X: Features
            y: True values
            exclude_closed: Whether to exclude closed rides
            closed_mask: Boolean mask for closed rides
            
        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        predictions = self.predict(X)
        
        # Filter closed rides if requested
        if exclude_closed and closed_mask is not None:
            open_mask = ~closed_mask
            y_eval = y[open_mask]
            predictions_eval = predictions[open_mask]
        else:
            y_eval = y
            predictions_eval = predictions
        
        # Calculate metrics
        mae = mean_absolute_error(y_eval, predictions_eval)
        rmse = np.sqrt(mean_squared_error(y_eval, predictions_eval))
        r2 = r2_score(y_eval, predictions_eval)
        
        # Calculate sMAPE
        non_zero_mask = y_eval > 0.1
        if np.sum(non_zero_mask) > 0:
            y_nz = y_eval[non_zero_mask]
            pred_nz = predictions_eval[non_zero_mask]
            numerator = np.abs(y_nz - pred_nz)
            denominator = np.abs(y_nz) + np.abs(pred_nz)
            smape = np.mean(numerator / denominator) * 100
        else:
            smape = 0.0
        
        metrics = {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "smape": smape,
            "n_samples": len(y_eval)
        }
        
        logger.info(f"Baseline metrics: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}, sMAPE={smape:.2f}%")
        
        return metrics
    
    def get_feature_importances(self):
        """Get feature importances from the model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importances")
        return self.model.feature_importances_
    
    def get_params(self):
        """Get model parameters"""
        return {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'random_state': self.random_state
        }