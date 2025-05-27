# Waitless Autoregressive TCN with Cached Scheduled Sampling
"""
A modular implementation of autoregressive TCN with cached scheduled sampling
for wait time prediction.
"""

__version__ = "1.0.0"
__author__ = "Waitless Team"

# Make key components easily accessible
from .models.tcn_model import AutoregressiveTCNModel
from .inference.predictor import WaitTimePredictor
from .training.trainer import CachedScheduledSamplingTCNTrainer

__all__ = [
    'AutoregressiveTCNModel',
    'WaitTimePredictor', 
    'CachedScheduledSamplingTCNTrainer'
]