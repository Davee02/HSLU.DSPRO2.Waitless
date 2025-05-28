import logging
import numpy as np
import torch
from torch.amp import autocast

logger = logging.getLogger(__name__)

class CachedScheduledSampling:
    def __init__(self, cache_update_frequency=5, max_cache_size=100000):
        self.prediction_cache = {}
        self.cache_epoch = -1
        self.cache_update_frequency = cache_update_frequency
        self.max_cache_size = max_cache_size
        
        # Diagnostic counters
        self.cache_stats = {
            'total_cache_attempts': 0,
            'successful_cache_entries': 0,
            'failed_cache_attempts': 0,
            'model_prediction_failures': 0,
            'lookup_attempts': 0,
            'lookup_hits': 0,
            'lookup_misses': 0
        }
        
        logger.info(f"Initialized cached scheduled sampling (update every {cache_update_frequency} epochs)")

    def should_update_cache(self, current_epoch):
        return current_epoch == 0 or (current_epoch > 0 and current_epoch % self.cache_update_frequency == 0)

    def update_cache(self, model, gb_model, dataset, device):
        if model is None or gb_model is None:
            logger.warning("Models not available for cache update")
            return

        logger.info(f"🔄 Starting cache update for epoch {dataset.current_epoch}...")
        logger.info(f"📊 Dataset info: {len(dataset.valid_indices)} valid indices, seq_length={dataset.seq_length}")
        
        # Reset diagnostic counters for this update
        update_stats = {
            'samples_processed': 0,
            'samples_with_cache': 0,
            'samples_without_cache': 0,
            'total_predictions_cached': 0,
            'prediction_failures': 0,
            'samples_expecting_cache': 0,  # NEW: Samples that should have cache
            'samples_with_successful_cache': 0,  # NEW: Samples that got cache
            'samples_with_failed_cache': 0,  # NEW: Samples that needed but didn't get cache
            'samples_no_cache_needed': 0  # NEW: Samples that legitimately don't need cache
        }
        
        self.prediction_cache.clear()
        model_was_training = model.training
        model.eval()

        batch_size = 8192
        cache_count = 0
        total_batches = (len(dataset.valid_indices) + batch_size - 1) // batch_size

        try:
            with torch.no_grad():
                for batch_num, start_idx in enumerate(range(0, len(dataset.valid_indices), batch_size)):
                    end_idx = min(start_idx + batch_size, len(dataset.valid_indices))
                    batch_indices = dataset.valid_indices[start_idx:end_idx]
                    
                    logger.info(f"📦 Processing batch {batch_num + 1}/{total_batches} ({len(batch_indices)} samples)...")

                    batch_cache_successes = 0
                    batch_cache_failures = 0
                    batch_no_cache_needed = 0
                    
                    for sample_idx in batch_indices:
                        if cache_count >= self.max_cache_size:
                            logger.warning(f"⚠️  Cache size limit reached ({self.max_cache_size})")
                            break

                        # Track cache attempt
                        update_stats['samples_processed'] += 1
                        self.cache_stats['total_cache_attempts'] += 1
                        
                        cache_result = self._cache_sample_predictions(
                            sample_idx, model, gb_model, dataset, device, update_stats
                        )
                        
                        cached_predictions, needs_cache = cache_result
                        
                        if needs_cache:
                            if cached_predictions > 0:
                                batch_cache_successes += 1
                                update_stats['samples_with_cache'] += 1
                                update_stats['total_predictions_cached'] += cached_predictions
                            else:
                                batch_cache_failures += 1
                                update_stats['samples_without_cache'] += 1
                        else:
                            batch_no_cache_needed += 1
                        
                        cache_count += 1

                    logger.info(f"📊 Batch {batch_num + 1} results:")
                    logger.info(f"   • Cache successes: {batch_cache_successes}")
                    logger.info(f"   • Cache failures: {batch_cache_failures}")
                    logger.info(f"   • No cache needed: {batch_no_cache_needed}")
                    
                    if cache_count >= self.max_cache_size:
                        break

        except Exception as e:
            logger.error(f"❌ Error updating cache: {e}")

        finally:
            if model_was_training:
                model.train()

        # Final cache analysis
        self._analyze_final_cache_structure()
        
        self.cache_epoch = dataset.current_epoch
        
        # Enhanced cache expectation analysis
        logger.info(f"📊 Cache Expectation Analysis:")
        logger.info(f"   • Total samples processed: {update_stats['samples_processed']}")
        logger.info(f"   • Samples expecting cache: {update_stats['samples_expecting_cache']}")
        logger.info(f"   • Samples with successful cache: {update_stats['samples_with_successful_cache']}")
        logger.info(f"   • Samples with failed cache: {update_stats['samples_with_failed_cache']}")
        logger.info(f"   • Samples not needing cache: {update_stats['samples_no_cache_needed']}")
        
        expected_cache = update_stats['samples_expecting_cache']
        successful_cache = update_stats['samples_with_successful_cache']
        if expected_cache > 0:
            cache_success_rate = (successful_cache / expected_cache) * 100
            logger.info(f"   • Cache success rate: {cache_success_rate:.1f}%")
        
        # Log comprehensive update results
        logger.info(f"✅ Cache update completed for epoch {self.cache_epoch}")
        logger.info(f"📊 Cache Update Stats:")
        logger.info(f"   • Total samples processed: {update_stats['samples_processed']}")
        logger.info(f"   • Samples with cache entries: {update_stats['samples_with_cache']}")
        logger.info(f"   • Samples without cache entries: {update_stats['samples_without_cache']}")
        logger.info(f"   • Total predictions cached: {update_stats['total_predictions_cached']}")
        logger.info(f"   • Prediction failures: {update_stats['prediction_failures']}")
        logger.info(f"   • Final cache size: {len(self.prediction_cache)} samples")
        logger.info(f"🔮 Next cache update scheduled for epoch {self.cache_epoch + self.cache_update_frequency}")

    def _cache_sample_predictions(self, target_idx, model, gb_model, dataset, device, update_stats):
        """
        Cache predictions for a single sample.
        Returns: (number_of_cached_predictions, needs_cache_flag)
        """
        target_date = dataset.dates.iloc[target_idx]
        sequence_start = target_idx - dataset.seq_length
        sequence_indices = list(range(sequence_start, target_idx))

        static_features = dataset.X_static[target_idx]
        sample_cache = {}
        current_sequence = []
        
        # Track what this sample actually needs
        same_day_positions_needing_cache = []
        cached_positions = []
        failed_positions = []

        for seq_position, seq_idx in enumerate(sequence_indices):
            seq_timestamp = dataset.timestamps.iloc[seq_idx]
            seq_date = dataset.dates.iloc[seq_idx]

            # Check if this position needs caching (same-day after opening)
            if (seq_date == target_date):
                same_day_positions_needing_cache.append(seq_position)
                
                try:
                    pred_wait_time, pred_residual = self._generate_model_prediction(
                        static_features, current_sequence, seq_idx, model, gb_model, dataset, device
                    )
                    sample_cache[seq_position] = (pred_wait_time, pred_residual)
                    current_sequence.extend([pred_wait_time, pred_residual])
                    cached_positions.append(seq_position)
                    
                except Exception as e:
                    logger.debug(f"❌ Failed to cache prediction for sample {target_idx}, seq {seq_position}: {e}")
                    failed_positions.append(seq_position)
                    update_stats['prediction_failures'] += 1
                    self.cache_stats['failed_cache_attempts'] += 1
                    
                    # Fallback to actual values for sequence building
                    actual_wait_time = dataset.wait_times[seq_idx]
                    actual_residual = dataset.residuals[seq_idx]
                    current_sequence.extend([actual_wait_time, actual_residual])
            else:
                # Use actual values for non-same-day data
                actual_wait_time = dataset.wait_times[seq_idx]
                actual_residual = dataset.residuals[seq_idx]
                current_sequence.extend([actual_wait_time, actual_residual])

        # Determine if this sample needs cache
        needs_cache = len(same_day_positions_needing_cache) > 0
        
        # Update expectation tracking
        if needs_cache:
            update_stats['samples_expecting_cache'] += 1
            if sample_cache:
                update_stats['samples_with_successful_cache'] += 1
            else:
                update_stats['samples_with_failed_cache'] += 1
        else:
            update_stats['samples_no_cache_needed'] += 1

        # Store cache if we have any successful predictions
        if sample_cache:
            self.prediction_cache[target_idx] = sample_cache
            self.cache_stats['successful_cache_entries'] += 1
        
        # Log problematic samples that SHOULD have cache but DON'T
        if same_day_positions_needing_cache and not sample_cache:
            logger.warning(f"⚠️ Sample {target_idx} NEEDS cache for positions {same_day_positions_needing_cache} but got NONE")
            logger.warning(f"   Target date: {target_date}, target time: {dataset.timestamps.iloc[target_idx]}")
            logger.warning(f"   All failed positions: {failed_positions}")
        
        # Log detailed info for first few samples or problematic cases
        if len(self.prediction_cache) <= 5 or (len(failed_positions) > 0 and len(cached_positions) == 0):
            logger.debug(f"🔍 Sample {target_idx} cache details:")
            logger.debug(f"   • Same-day positions needing cache: {same_day_positions_needing_cache}")
            logger.debug(f"   • Successfully cached: {cached_positions}")
            logger.debug(f"   • Failed positions: {failed_positions}")
            logger.debug(f"   • Cache entry keys: {list(sample_cache.keys())}")
            logger.debug(f"   • Needs cache: {needs_cache}")
        
        return len(sample_cache), needs_cache

    def _generate_model_prediction(self, static_features, sequence_features, target_idx, 
                                   model, gb_model, dataset, device):
        try:
            expected_sequence_size = dataset.seq_length * 2
            current_sequence = np.array(sequence_features, dtype=np.float32)

            if len(current_sequence) < expected_sequence_size:
                padding_needed = expected_sequence_size - len(current_sequence)
                current_sequence = np.concatenate([
                    np.zeros(padding_needed, dtype=np.float32), 
                    current_sequence
                ])
            elif len(current_sequence) > expected_sequence_size:
                current_sequence = current_sequence[-expected_sequence_size:]

            combined_features = np.concatenate([static_features, current_sequence])
            model_input = torch.FloatTensor(combined_features).unsqueeze(0).to(device)

            with autocast(device_type="cuda"):
                residual_pred = model(model_input)[0, 0].item()

            gb_pred = gb_model.predict(static_features.reshape(1, -1))[0]
            combined_pred = gb_pred + residual_pred

            wait_noise_std = max(0.05, abs(combined_pred) * 0.1)
            residual_noise_std = max(0.025, abs(residual_pred) * 0.1)

            noisy_wait_pred = combined_pred + np.random.normal(0, wait_noise_std)
            noisy_residual_pred = residual_pred + np.random.normal(0, residual_noise_std)

            return max(0, noisy_wait_pred), noisy_residual_pred

        except Exception as e:
            logger.debug(f"❌ Model prediction failed during caching for target_idx {target_idx}: {e}")
            self.cache_stats['model_prediction_failures'] += 1
            
            # Fallback to noisy actual values
            actual_wait_time = dataset.wait_times[target_idx]
            actual_residual = dataset.residuals[target_idx]

            wait_noise_std = max(0.1, actual_wait_time * 0.15)
            residual_noise_std = max(0.05, abs(actual_residual) * 0.2)

            simulated_wait_time = actual_wait_time + np.random.normal(0, wait_noise_std)
            simulated_residual = actual_residual + np.random.normal(0, residual_noise_std)

            return max(0, simulated_wait_time), simulated_residual

    def _analyze_final_cache_structure(self):
        """Analyze the structure of the final cache"""
        if not self.prediction_cache:
            logger.warning("⚠️  Cache is empty after update!")
            return
        
        cache_sizes = [len(sample_cache) for sample_cache in self.prediction_cache.values()]
        
        logger.info(f"📊 Cache Structure Analysis:")
        logger.info(f"   • Total cached samples: {len(self.prediction_cache)}")
        logger.info(f"   • Min predictions per sample: {min(cache_sizes)}")
        logger.info(f"   • Max predictions per sample: {max(cache_sizes)}")
        logger.info(f"   • Avg predictions per sample: {np.mean(cache_sizes):.1f}")
        
        # Show distribution of cache sizes
        from collections import Counter
        size_distribution = Counter(cache_sizes)
        logger.info(f"   • Cache size distribution: {dict(size_distribution)}")
        
        # Sample a few cache entries for detailed inspection
        sample_keys = list(self.prediction_cache.keys())[:3]
        for idx in sample_keys:
            positions = list(self.prediction_cache[idx].keys())
            logger.debug(f"   • Sample {idx} has cached positions: {sorted(positions)}")

    def get_cached_prediction(self, sample_idx, seq_position):
        """Get cached prediction with detailed logging"""
        self.cache_stats['lookup_attempts'] += 1
        
        # Check if sample exists in cache
        if sample_idx not in self.prediction_cache:
            self.cache_stats['lookup_misses'] += 1
            logger.debug(f"🔍 Cache MISS: sample_idx {sample_idx} not in cache")
            logger.debug(f"   • Cache has {len(self.prediction_cache)} samples")
            logger.debug(f"   • Cache epoch: {self.cache_epoch}")
            return None
        
        # Check if sequence position exists for this sample
        sample_cache = self.prediction_cache[sample_idx]
        if seq_position not in sample_cache:
            self.cache_stats['lookup_misses'] += 1
            logger.debug(f"🔍 Cache MISS: seq_position {seq_position} not cached for sample {sample_idx}")
            logger.debug(f"   • Available positions for sample {sample_idx}: {sorted(sample_cache.keys())}")
            return None
        
        # Cache hit!
        self.cache_stats['lookup_hits'] += 1
        prediction = sample_cache[seq_position]
        
        # Log cache hits less frequently to reduce noise
        if self.cache_stats['lookup_hits'] % 100 == 1:  # Log every 100th hit
            logger.debug(f"✅ Cache HIT: sample {sample_idx}, position {seq_position}, prediction: {prediction}")
        
        return prediction

    def log_cache_statistics(self):
        """Log overall cache usage statistics"""
        stats = self.cache_stats
        
        total_lookups = stats['lookup_attempts']
        if total_lookups > 0:
            hit_rate = (stats['lookup_hits'] / total_lookups) * 100
            miss_rate = (stats['lookup_misses'] / total_lookups) * 100
        else:
            hit_rate = miss_rate = 0
        
        total_attempts = stats['total_cache_attempts']
        if total_attempts > 0:
            success_rate = (stats['successful_cache_entries'] / total_attempts) * 100
        else:
            success_rate = 0
        
        logger.info(f"📊 Overall Cache Statistics:")
        logger.info(f"   • Total cache update attempts: {total_attempts}")
        logger.info(f"   • Successful cache entries: {stats['successful_cache_entries']} ({success_rate:.1f}%)")
        logger.info(f"   • Failed cache attempts: {stats['failed_cache_attempts']}")
        logger.info(f"   • Model prediction failures: {stats['model_prediction_failures']}")
        logger.info(f"   • Cache lookup attempts: {total_lookups}")
        logger.info(f"   • Cache hits: {stats['lookup_hits']} ({hit_rate:.1f}%)")
        logger.info(f"   • Cache misses: {stats['lookup_misses']} ({miss_rate:.1f}%)")

    def clear_cache(self):
        self.prediction_cache.clear()
        logger.info("🧹 Prediction cache cleared")
        
        # Log final statistics before clearing
        self.log_cache_statistics()