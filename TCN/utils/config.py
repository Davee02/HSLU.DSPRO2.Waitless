import yaml
import json
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file"""
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            return yaml.safe_load(f)
        else:
            return json.load(f)


def create_config_from_ride(ride_name: str, rides_config_path: str = "TCN/configs/rides_config.yaml") -> Dict[str, Any]:
    """Create a training config from ride name and rides config"""
    with open(rides_config_path, 'r') as f:
        rides_config = yaml.safe_load(f)
    
    if ride_name not in rides_config['rides']:
        raise ValueError(f"Ride '{ride_name}' not found in {rides_config_path}")
    
    # Start with default parameters
    config = rides_config['default_params'].copy()
    
    # Add global settings
    config.update(rides_config['global_settings'])
    
    # Add ride-specific settings
    ride_info = rides_config['rides'][ride_name]
    config['data_path'] = ride_info['data_path']
    config['target_ride'] = ride_name
    
    # Autoregressive-specific defaults with cached scheduled sampling
    autoregressive_defaults = {
        'seq_length': 96,        # 96 * 5min
        'batch_size': 1024,
        'num_channels': 256,
        'kernel_size': 3,
        'num_layers': 8,
        'dropout': 0.3,
        'learning_rate': 0.0005,
        'weight_decay': 1e-5,
        'opening_hour': 9,
        'closing_hour': 21,
        # GradientBoosting parameters
        'gb_n_estimators': 100,
        'gb_learning_rate': 0.1,
        'gb_max_depth': 6,
        'gb_min_samples_split': 10,
        'gb_min_samples_leaf': 5,
        # Cached scheduled sampling parameters
        'sampling_strategy': 'linear',  # 'linear', 'exponential', 'inverse_sigmoid'
        'noise_factor': 0.15,           # Standard deviation factor for prediction noise
        'cache_update_frequency': 5,     # Update cache every N epochs
        'max_cache_size': 100000,       # Maximum cached predictions
        'use_torch_compile': True,      # Enable torch.compile
        'use_mixed_precision': True,    # Enable mixed precision
        'run_name': 'cached_scheduled_sampling_autoregressive'
    }
    
    for key, value in autoregressive_defaults.items():
        if key not in config:
            config[key] = value
    
    return config


def save_config(config: Dict[str, Any], path: str) -> None:
    """Save configuration to file"""
    with open(path, 'w') as f:
        if path.endswith('.yaml') or path.endswith('.yml'):
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        else:
            json.dump(config, f, indent=2)


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configurations, with override taking precedence"""
    merged = base_config.copy()
    merged.update(override_config)
    return merged