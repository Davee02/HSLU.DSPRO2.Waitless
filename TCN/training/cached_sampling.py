import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)


class CachedScheduledSampling:
    """
    Manages cached predictions for scheduled sampling to avoid dynamic shapes
    and enable torch.compile compatibility.
    """
    def __init__(self, cache_update_frequency=5, max_cache_size=100000):
        """
        Args:
            cache_update_frequency: Update cache every N epochs
            max_cache_size: Maximum number of cached predictions
        """
        self.prediction_cache = {}  # {sample_idx: {timestep: (wait_pred, residual_pred)}}
        self.cache_epoch = -1
        self.cache_update_frequency = cache_update_frequency
        self.max_cache_size = max_cache_size
        logger.info(f"Initialized cached scheduled sampling (update every {cache_update_frequency} epochs)")
    
    def should_update_cache(self, current_epoch):
        """Check if cache should be updated"""
        # Update cache at epoch 0 and then every cache_update_frequency epochs
        if current_epoch == 0:
            return True
        elif current_epoch > 0 and current_epoch % self.cache_update_frequency == 0:
            return True
        else:
            return False
    
    def update_cache(self, model, gb_model, dataset, device):
        """
        Pre-compute model predictions for all relevant sample-timestep combinations.
        This runs in batches to avoid memory issues.
        """
        if model is None or gb_model is None:
            logger.warning("Models not available for cache update")
            return
        
        logger.info(f"Updating prediction cache for epoch {dataset.current_epoch}...")
        
        # Clear old cache
        self.prediction_cache.clear()
        
        # Set model to evaluation mode
        model_was_training = model.training
        model.eval()
        
        batch_size = 256  # Smaller batch size for cache update
        cache_count = 0
        
        try:
            with torch.no_grad():
                # Process samples in batches for memory efficiency
                for start_idx in range(0, len(dataset.valid_indices), batch_size):
                    end_idx = min(start_idx + batch_size, len(dataset.valid_indices))
                    batch_indices = dataset.valid_indices[start_idx:end_idx]
                    
                    for sample_idx in batch_indices:
                        if cache_count >= self.max_cache_size:
                            logger.warning(f"Cache size limit reached ({self.max_cache_size})")
                            break
                        
                        # Pre-compute predictions for this sample's autoregressive sequence
                        self._cache_sample_predictions(
                            sample_idx, model, gb_model, dataset, device
                        )
                        cache_count += 1
                    
                    if cache_count >= self.max_cache_size:
                        break
        
        except Exception as e:
            logger.error(f"Error updating cache: {e}")
        
        finally:
            # Restore model training state
            if model_was_training:
                model.train()
        
        # Update cache epoch to current epoch
        self.cache_epoch = dataset.current_epoch
        logger.info(f"Cache updated with {len(self.prediction_cache)} samples for epoch {self.cache_epoch}")
        logger.info(f"Next cache update scheduled for epoch {self.cache_epoch + self.cache_update_frequency}")
    
    def _cache_sample_predictions(self, target_idx, model, gb_model, dataset, device):
        """Cache predictions for a single sample's autoregressive sequence"""
        target_timestamp = dataset.timestamps.iloc[target_idx]
        target_date = dataset.dates.iloc[target_idx]
        
        sequence_start = target_idx - dataset.seq_length
        sequence_indices = list(range(sequence_start, target_idx))
        
        static_features = dataset.X_static[target_idx]
        sample_cache = {}
        
        # Build sequence incrementally and cache predictions at each step
        current_sequence = []
        
        for seq_position, seq_idx in enumerate(sequence_indices):
            seq_timestamp = dataset.timestamps.iloc[seq_idx]
            seq_date = dataset.dates.iloc[seq_idx]
            
            # Only cache predictions for same-day future predictions
            if (seq_date == target_date and 
                seq_timestamp.hour >= dataset.opening_hour):
                
                try:
                    # Predict based on current sequence state
                    pred_wait_time, pred_residual = self._generate_model_prediction(
                        static_features, current_sequence, seq_idx, model, gb_model, dataset, device
                    )
                    
                    # Cache the prediction
                    sample_cache[seq_position] = (pred_wait_time, pred_residual)
                    
                    # Add prediction to sequence for next iteration
                    current_sequence.extend([pred_wait_time, pred_residual])
                    
                except Exception as e:
                    logger.debug(f"Failed to cache prediction for sample {target_idx}, seq {seq_position}: {e}")
                    # Add actual values as fallback
                    actual_wait_time = dataset.wait_times[seq_idx]
                    actual_residual = dataset.residuals[seq_idx]
                    current_sequence.extend([actual_wait_time, actual_residual])
            else:
                # Add actual historical values (not same day)
                actual_wait_time = dataset.wait_times[seq_idx]
                actual_residual = dataset.residuals[seq_idx]
                current_sequence.extend([actual_wait_time, actual_residual])
        
        # Store cache for this sample
        if sample_cache:
            self.prediction_cache[target_idx] = sample_cache
    
    def _generate_model_prediction(self, static_features, sequence_features, target_idx, 
                                   model, gb_model, dataset, device):
        """Generate model prediction for caching"""
        try:
            # Prepare sequence for model input
            expected_sequence_size = dataset.seq_length * 2
            current_sequence = np.array(sequence_features, dtype=np.float32)
            
            if len(current_sequence) < expected_sequence_size:
                # Pad with zeros if sequence too short
                padding_needed = expected_sequence_size - len(current_sequence)
                current_sequence = np.concatenate([
                    np.zeros(padding_needed, dtype=np.float32), 
                    current_sequence
                ])
            elif len(current_sequence) > expected_sequence_size:
                # Truncate if too long
                current_sequence = current_sequence[-expected_sequence_size:]
            
            # Combine features for model input
            combined_features = np.concatenate([static_features, current_sequence])
            model_input = torch.FloatTensor(combined_features).unsqueeze(0).to(device)
            
            # Get residual prediction
            residual_pred = model(model_input).cpu().numpy().flatten()[0]
            
            # Get baseline prediction
            gb_pred = gb_model.predict(static_features.reshape(1, -1))[0]
            
            # Combined prediction
            combined_pred = gb_pred + residual_pred
            
            # Add small amount of noise for training diversity
            wait_noise_std = max(0.05, abs(combined_pred) * 0.1)
            residual_noise_std = max(0.025, abs(residual_pred) * 0.1)
            
            noisy_wait_pred = combined_pred + np.random.normal(0, wait_noise_std)
            noisy_residual_pred = residual_pred + np.random.normal(0, residual_noise_std)
            
            return max(0, noisy_wait_pred), noisy_residual_pred
            
        except Exception as e:
            logger.debug(f"Model prediction failed during caching: {e}")
            # Fallback to noisy actual values
            actual_wait_time = dataset.wait_times[target_idx]
            actual_residual = dataset.residuals[target_idx]
            
            wait_noise_std = max(0.1, actual_wait_time * 0.15)
            residual_noise_std = max(0.05, abs(actual_residual) * 0.2)
            
            simulated_wait_time = actual_wait_time + np.random.normal(0, wait_noise_std)
            simulated_residual = actual_residual + np.random.normal(0, residual_noise_std)
            
            return max(0, simulated_wait_time), simulated_residual
    
    def get_cached_prediction(self, sample_idx, seq_position):
        """Get cached prediction for a specific sample and sequence position"""
        if sample_idx in self.prediction_cache:
            if seq_position in self.prediction_cache[sample_idx]:
                return self.prediction_cache[sample_idx][seq_position]
        return None
    
    def clear_cache(self):
        """Clear the prediction cache"""
        self.prediction_cache.clear()
        logger.info("Prediction cache cleared")