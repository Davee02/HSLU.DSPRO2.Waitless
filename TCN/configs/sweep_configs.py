"""
Sweep configuration templates for different hyperparameter search strategies
"""

SWEEP_CONFIGS = {
    'standard': {
        'method': 'bayes',
        'metric': {
            'name': 'val_loss',
            'goal': 'minimize'
        },
        'parameters': {
            # Fixed parameters
            'splits_output_dir': {'value': "data/processed/splits"},
            'epochs': {'value': 60},
            'patience': {'value': 20},
            'seed': {'value': 42},
            'use_wandb': {'value': True},
            'use_torch_compile': {'value': True},
            'use_mixed_precision': {'value': True},
            'set_float32_matmul_precision': {'value': True},
            
            # TCN architecture parameters
            'seq_length': {'values': [324]}, # 324 would be 27 hours
            'batch_size': {'values': [1024]},
            'num_channels': {'values': [128, 256]},
            'kernel_size': {'values': [3, 5, 7]},
            'num_layers': {'values': [6, 8, 10]},
            'dropout': {'values': [0.1, 0.2, 0.3, 0.4]},
            'learning_rate': {'values': [0.001, 0.0001, 0.00001]},
            'weight_decay': {'values': [1e-5, 1e-4, 1e-3]},
            
            # Cached scheduled sampling parameters
            'sampling_strategy': {'values': ['linear']},
            'noise_factor': {'values': [0.15, 0.2]},
            'cache_update_frequency': {'values': [5]},
            
            # GradientBoosting parameters
            'gb_n_estimators': {'values': [100, 150, 200]},
            'gb_max_depth': {'values': [4, 6, 8]},
            'gb_learning_rate': {'values': [0.05, 0.1, 0.15]},
        }
    },
    
    'comprehensive': {
        'method': 'bayes',
        'metric': {
            'name': 'test_mae',
            'goal': 'minimize'
        },
        'parameters': {
            # Fixed parameters
            'splits_output_dir': {'value': "data/processed/splits"},
            'epochs': {'value': 200},
            'patience': {'value': 30},
            'seed': {'value': 42},
            'use_wandb': {'value': True},
            'use_torch_compile': {'value': True},
            'use_mixed_precision': {'value': True},
            
            # TCN architecture - wider search
            'seq_length': {'values': [48, 96, 192, 324]},
            'batch_size': {'values': [64, 128, 256, 512, 1024]},
            'num_channels': {'values': [64, 128, 256, 512, 1024]},
            'kernel_size': {'values': [2, 3, 4, 5, 7, 9]},
            'num_layers': {'values': [4, 6, 8, 10, 12]},
            'dropout': {'distribution': 'uniform', 'min': 0.0, 'max': 0.5},
            'learning_rate': {'distribution': 'log_uniform_values', 'min': 1e-4, 'max': 1e-2},
            'weight_decay': {'distribution': 'log_uniform_values', 'min': 1e-6, 'max': 1e-2},
            
            # Cached scheduled sampling
            'sampling_strategy': {'values': ['linear', 'exponential', 'inverse_sigmoid']},
            'noise_factor': {'distribution': 'uniform', 'min': 0.05, 'max': 0.3},
            'cache_update_frequency': {'values': [1, 3, 5, 7, 10]},
            
            # GradientBoosting
            'gb_n_estimators': {'values': [50, 100, 150, 200, 300]},
            'gb_max_depth': {'values': [3, 4, 5, 6, 7, 8, 10]},
            'gb_learning_rate': {'distribution': 'uniform', 'min': 0.01, 'max': 0.3},
            'gb_min_samples_split': {'values': [5, 10, 20, 50]},
            'gb_min_samples_leaf': {'values': [1, 5, 10, 20]},
        }
    },
    
    'quick': {
        'method': 'grid',
        'metric': {
            'name': 'test_mae',
            'goal': 'minimize'
        },
        'parameters': {
            # Fixed parameters
            'splits_output_dir': {'value': "data/processed/splits"},
            'epochs': {'value': 50},
            'patience': {'value': 10},
            'seed': {'value': 42},
            'use_wandb': {'value': True},
            'use_torch_compile': {'value': True},
            'use_mixed_precision': {'value': True},
            
            # Limited search for quick experiments
            'seq_length': {'values': [96, 192]},
            'batch_size': {'values': [128, 256]},
            'num_channels': {'values': [128, 256]},
            'kernel_size': {'values': [3, 5]},
            'num_layers': {'value': 8},
            'dropout': {'values': [0.2, 0.3]},
            'learning_rate': {'values': [0.001, 0.0005]},
            'weight_decay': {'value': 1e-5},
            
            # Fixed cached scheduled sampling
            'sampling_strategy': {'value': 'linear'},
            'noise_factor': {'value': 0.15},
            'cache_update_frequency': {'value': 5},
            
            # Fixed GradientBoosting
            'gb_n_estimators': {'value': 100},
            'gb_max_depth': {'value': 6},
            'gb_learning_rate': {'value': 0.1},
        }
    },
    
    'scheduler_focused': {
        'method': 'bayes',
        'metric': {
            'name': 'test_mae',
            'goal': 'minimize'
        },
        'parameters': {
            # Fixed parameters
            'splits_output_dir': {'value': "data/processed/splits"},
            'epochs': {'value': 150},
            'patience': {'value': 25},
            'seed': {'value': 42},
            'use_wandb': {'value': True},
            
            # Fixed architecture (use best from previous sweeps)
            'seq_length': {'value': 192},
            'batch_size': {'value': 256},
            'num_channels': {'value': 256},
            'kernel_size': {'value': 3},
            'num_layers': {'value': 8},
            'dropout': {'value': 0.3},
            'weight_decay': {'value': 1e-5},
            
            # Learning rate and scheduler exploration
            'learning_rate': {'distribution': 'log_uniform_values', 'min': 1e-4, 'max': 1e-2},
            'scheduler_type': {'values': ['OneCycle', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts', 'StepLR']},
            
            # OneCycle parameters
            'pct_start': {'values': [0.1, 0.2, 0.3]},
            
            # CosineAnnealingWarmRestarts parameters
            't_0': {'values': [10, 15, 20, 25]},
            't_mult': {'values': [1, 2]},
            'eta_min': {'distribution': 'log_uniform_values', 'min': 1e-7, 'max': 1e-4},
            
            # StepLR parameters
            'step_size': {'values': [10, 20, 30]},
            'gamma': {'values': [0.1, 0.5, 0.7]},
            
            # Sampling strategy focus
            'sampling_strategy': {'values': ['linear', 'exponential', 'inverse_sigmoid']},
            'noise_factor': {'distribution': 'uniform', 'min': 0.05, 'max': 0.25},
            'cache_update_frequency': {'values': [3, 5, 7]},
            
            # Fixed GradientBoosting
            'gb_n_estimators': {'value': 100},
            'gb_max_depth': {'value': 6},
            'gb_learning_rate': {'value': 0.1},
        }
    }
}


def get_sweep_config(config_name: str = 'standard') -> dict:
    """Get a sweep configuration by name"""
    if config_name not in SWEEP_CONFIGS:
        raise ValueError(f"Unknown sweep config: {config_name}. Available: {list(SWEEP_CONFIGS.keys())}")
    return SWEEP_CONFIGS[config_name].copy()


def list_sweep_configs() -> list:
    """List available sweep configurations"""
    return list(SWEEP_CONFIGS.keys())


def create_custom_sweep_config(base_config: str = 'standard', **overrides) -> dict:
    """Create a custom sweep config based on a template with overrides"""
    config = get_sweep_config(base_config)
    
    # Apply overrides
    for key, value in overrides.items():
        if key in config['parameters']:
            config['parameters'][key] = value
        else:
            # Add new parameter
            config['parameters'][key] = value
    
    return config