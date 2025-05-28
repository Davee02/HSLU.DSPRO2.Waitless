import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .data_utils import calculate_scheduled_sampling_probability

logger = logging.getLogger(__name__)


class CachedScheduledSamplingDataset(Dataset):
    """
    Dataset for autoregressive residual prediction with cached scheduled sampling.
    Uses pre-computed predictions to avoid dynamic shapes during training.
    """
    def __init__(self, X_static, residuals, wait_times, seq_length, timestamps, 
                 opening_hour=11, closing_hour=17, current_epoch=0, total_epochs=100,
                 sampling_strategy="linear", noise_factor=0.15, 
                 prediction_cache=None):
        """
        Args:
            prediction_cache: CachedScheduledSampling instance
        """
        self.X_static = X_static
        self.residuals = residuals
        self.wait_times = wait_times
        self.seq_length = seq_length
        self.current_epoch = current_epoch
        self.total_epochs = total_epochs
        self.sampling_strategy = sampling_strategy
        self.noise_factor = noise_factor
        self.prediction_cache = prediction_cache
        
        if isinstance(timestamps, pd.Series):
            self.timestamps = pd.to_datetime(timestamps.reset_index(drop=True))
        else:
            self.timestamps = pd.to_datetime(timestamps)
        
        self.opening_hour = opening_hour
        self.closing_hour = closing_hour
        
        # For same-day logic
        self.dates = self.timestamps.dt.date
        
        self.valid_indices = self._create_valid_indices()
        
        # Calculate scheduled sampling probability
        self.teacher_forcing_prob = self._calculate_teacher_forcing_prob()
        
        # Enhanced diagnostic tracking
        self._sequence_stats = {
            'total_sequences_generated': 0,
            'total_same_day_positions': 0,
            'teacher_forcing_positions': 0,
            'scheduled_sampling_positions': 0,
            'cached_prediction_positions': 0,
            'fallback_positions': 0,
            'samples_with_cache_needs': 0,
            'samples_with_zero_cache_hits': 0,
            'cache_hit_distribution': {}  # seq_position -> hit_count
        }
        
        logger.info(f"🔄 Cached Scheduled Sampling Dataset initialized:")
        logger.info(f"   • Valid sequences: {len(self.valid_indices)}")
        logger.info(f"   • Sequence length: {seq_length}")
        logger.info(f"   • Operating hours: {opening_hour}:00 - {closing_hour}:00")
        logger.info(f"   • Current epoch: {current_epoch}")
        logger.info(f"   • Teacher forcing prob: {self.teacher_forcing_prob:.3f}")
    
    def update_epoch(self, epoch):
        """Update current epoch and recalculate teacher forcing probability"""
        old_prob = self.teacher_forcing_prob
        self.current_epoch = epoch
        self.teacher_forcing_prob = self._calculate_teacher_forcing_prob()
        
        logger.info(f"📅 Epoch updated: {epoch}")
        logger.info(f"   • Teacher forcing prob: {old_prob:.3f} → {self.teacher_forcing_prob:.3f}")
        
        # Reset sequence stats for new epoch
        self._sequence_stats = {k: 0 if isinstance(v, int) else {} for k, v in self._sequence_stats.items()}
    
    def _calculate_teacher_forcing_prob(self):
        """Calculate teacher forcing probability based on current epoch and strategy"""
        return calculate_scheduled_sampling_probability(
            self.current_epoch, self.total_epochs, self.sampling_strategy
        )
    
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
        Create sequence with cached scheduled sampling.
        Uses pre-computed predictions for performance and compatibility.
        """
        target_timestamp = self.timestamps.iloc[target_idx]
        target_date = self.dates.iloc[target_idx]
        
        sequence_start = target_idx - self.seq_length
        sequence_indices = list(range(sequence_start, target_idx))
        
        static_features = self.X_static[target_idx]
        autoregressive_features = []
        
        # Enhanced tracking for this specific sequence
        sequence_analysis = {
            'target_idx': target_idx,
            'target_date': target_date,
            'target_time': target_timestamp.strftime('%Y-%m-%d %H:%M'),
            'teacher_forcing_positions': [],
            'scheduled_sampling_positions': [],
            'cached_hit_positions': [],
            'cached_miss_positions': [],
            'historical_positions': []
        }
        
        self._sequence_stats['total_sequences_generated'] += 1
        
        for seq_position, seq_idx in enumerate(sequence_indices):
            seq_timestamp = self.timestamps.iloc[seq_idx]
            seq_date = self.dates.iloc[seq_idx]
            
            actual_wait_time = self.wait_times[seq_idx]
            actual_residual = self.residuals[seq_idx]
            
            # Same-day prediction logic with enhanced logging
            if seq_date == target_date:
                self._sequence_stats['total_same_day_positions'] += 1
                
                # Probabilistic scheduled sampling decision
                use_teacher_forcing = np.random.random() < self.teacher_forcing_prob
                
                if use_teacher_forcing:
                    # Teacher Forcing: Use actual ground truth values
                    autoregressive_features.extend([actual_wait_time, actual_residual])
                    sequence_analysis['teacher_forcing_positions'].append(seq_position)
                    self._sequence_stats['teacher_forcing_positions'] += 1
                    
                else:
                    # Scheduled Sampling: Try to use cached predictions
                    sequence_analysis['scheduled_sampling_positions'].append(seq_position)
                    self._sequence_stats['scheduled_sampling_positions'] += 1
                    
                    cached_pred = None
                    if self.prediction_cache:
                        cached_pred = self.prediction_cache.get_cached_prediction(target_idx, seq_position)
                    
                    if cached_pred is not None:
                        # Cache hit - use cached prediction
                        pred_wait_time, pred_residual = cached_pred
                        autoregressive_features.extend([pred_wait_time, pred_residual])
                        sequence_analysis['cached_hit_positions'].append(seq_position)
                        self._sequence_stats['cached_prediction_positions'] += 1
                        
                        # Track cache hit distribution
                        if seq_position not in self._sequence_stats['cache_hit_distribution']:
                            self._sequence_stats['cache_hit_distribution'][seq_position] = 0
                        self._sequence_stats['cache_hit_distribution'][seq_position] += 1
                        
                        # Log successful cache usage (less frequently to reduce noise)
                        if self._sequence_stats['cached_prediction_positions'] % 500 == 1:
                            logger.debug(f"✅ Cache hit for target {target_idx}, seq_pos {seq_position}")
                        
                    else:
                        # Cache miss - fallback to noisy actual values
                        sequence_analysis['cached_miss_positions'].append(seq_position)
                        self._sequence_stats['fallback_positions'] += 1
                        
                        wait_noise_std = max(0.1, actual_wait_time * self.noise_factor)
                        residual_noise_std = max(0.05, abs(actual_residual) * 0.2)
                        
                        simulated_wait_time = actual_wait_time + np.random.normal(0, wait_noise_std)
                        simulated_residual = actual_residual + np.random.normal(0, residual_noise_std)
                        simulated_wait_time = max(0, simulated_wait_time)
                        
                        autoregressive_features.extend([simulated_wait_time, simulated_residual])
                        
                        # Log cache misses for analysis
                        logger.debug(f"❌ Cache miss: target {target_idx}, seq_pos {seq_position} - fallback to noisy actual")
                        
            else:
                # Use actual historical values (different days)
                autoregressive_features.extend([actual_wait_time, actual_residual])
                sequence_analysis['historical_positions'].append(seq_position)
        
        # Analyze this sequence's cache dependency
        self._analyze_sequence_cache_usage(sequence_analysis)
        
        # Update high-level statistics
        if len(sequence_analysis['scheduled_sampling_positions']) > 0:
            self._sequence_stats['samples_with_cache_needs'] += 1
            if len(sequence_analysis['cached_hit_positions']) == 0:
                self._sequence_stats['samples_with_zero_cache_hits'] += 1
        
        # Periodic detailed logging
        if self._sequence_stats['total_sequences_generated'] % 1000 == 0:
            self._log_comprehensive_stats()
        
        # Combine static and autoregressive features
        combined_features = np.concatenate([
            static_features,
            autoregressive_features
        ])
        
        # Target is the actual residual at target_idx
        target_residual = self.residuals[target_idx]
        
        return combined_features.astype(np.float32), np.array([target_residual], dtype=np.float32)
    
    def _analyze_sequence_cache_usage(self, analysis):
        """Analyze cache usage patterns for a single sequence"""
        target_idx = analysis['target_idx']
        
        total_same_day = len(analysis['teacher_forcing_positions']) + len(analysis['scheduled_sampling_positions'])
        cache_dependent = len(analysis['scheduled_sampling_positions'])
        cache_hits = len(analysis['cached_hit_positions'])
        cache_misses = len(analysis['cached_miss_positions'])
        
        # Log problematic sequences
        if cache_dependent > 0 and cache_hits == 0:
            logger.debug(f"⚠️  Zero cache hits for target {target_idx}:")
            logger.debug(f"   • Needed cache for positions: {analysis['scheduled_sampling_positions']}")
            logger.debug(f"   • All cache misses at positions: {analysis['cached_miss_positions']}")
            logger.debug(f"   • Target time: {analysis['target_time']}")
        
        # Log sequences with mixed success (some hits, some misses)
        elif cache_hits > 0 and cache_misses > 0:
            logger.debug(f"🔄 Mixed cache results for target {target_idx}:")
            logger.debug(f"   • Cache hits at positions: {analysis['cached_hit_positions']}")
            logger.debug(f"   • Cache misses at positions: {analysis['cached_miss_positions']}")
    
    def _log_comprehensive_stats(self):
        """Log comprehensive statistics every N sequences"""
        stats = self._sequence_stats
        
        total_sequences = stats['total_sequences_generated']
        total_positions = stats['total_same_day_positions']
        
        if total_positions > 0:
            tf_rate = (stats['teacher_forcing_positions'] / total_positions) * 100
            ss_rate = (stats['scheduled_sampling_positions'] / total_positions) * 100
            cache_rate = (stats['cached_prediction_positions'] / stats['scheduled_sampling_positions']) * 100 if stats['scheduled_sampling_positions'] > 0 else 0
            fallback_rate = (stats['fallback_positions'] / stats['scheduled_sampling_positions']) * 100 if stats['scheduled_sampling_positions'] > 0 else 0
        else:
            tf_rate = ss_rate = cache_rate = fallback_rate = 0
        
        cache_needy_samples = stats['samples_with_cache_needs']
        zero_hit_samples = stats['samples_with_zero_cache_hits']
        zero_hit_rate = (zero_hit_samples / cache_needy_samples) * 100 if cache_needy_samples > 0 else 0

        # Log cache hit distribution by sequence position
        if stats['cache_hit_distribution']:
            sorted_positions = sorted(stats['cache_hit_distribution'].items())
            hit_summary = {pos: count for pos, count in sorted_positions}
    
    def log_final_statistics(self):
        """Log final comprehensive statistics"""
        logger.info(f"📊 FINAL Dataset Statistics:")
        self._log_comprehensive_stats()
        
        # Log cache performance from the cache object if available
        if self.prediction_cache:
            self.prediction_cache.log_cache_statistics()
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        target_idx = self.valid_indices[idx]
        X, y = self._get_autoregressive_sequence(target_idx)
        return torch.FloatTensor(X), torch.FloatTensor(y)