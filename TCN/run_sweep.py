#!/usr/bin/env python3
"""
Hyperparameter sweep script for TCN models
"""

import argparse
import yaml
import wandb
import copy
from train import TCNTrainer, load_config

def load_rides_config(rides_config_path="configs/rides_config.yaml"):
    """Load ride-specific configuration"""
    with open(rides_config_path, 'r') as f:
        return yaml.safe_load(f)

def create_sweep_config(base_sweep_config, ride_name, rides_config):
    """Create ride-specific sweep config from base config"""
    if ride_name not in rides_config['rides']:
        raise ValueError(f"Ride '{ride_name}' not found in rides_config.yaml. Available rides: {list(rides_config['rides'].keys())}")
    
    # Deep copy the base config
    sweep_config = copy.deepcopy(base_sweep_config)
    
    # Get ride-specific settings
    ride_info = rides_config['rides'][ride_name]
    global_settings = rides_config['global_settings']
    default_params = rides_config['default_params']
    
    # Add ride-specific parameters
    sweep_config['parameters']['data_path'] = {'value': ride_info['data_path']}
    sweep_config['parameters']['target_ride'] = {'value': ride_name}
    
    # Add global settings
    for key, value in global_settings.items():
        if key != 'base_data_dir':  # Skip base_data_dir as we use full paths
            sweep_config['parameters'][key] = {'value': value}
    
    # Add default parameters
    for key, value in default_params.items():
        if key not in sweep_config['parameters']:
            sweep_config['parameters'][key] = {'value': value}
    
    return sweep_config

def train_with_wandb():
    """Training function called by wandb agent"""
    run = wandb.init()
    config = dict(wandb.config)
    
    trainer = TCNTrainer(config)
    metrics = trainer.train_single_model()
    
    wandb.finish()
    return metrics

def setup_sweep(base_sweep_config_path, ride_name, project_name, entity, rides_config_path="configs/rides_config.yaml"):
    """Setup and start hyperparameter sweep for a specific ride"""
    
    # Load configurations
    with open(base_sweep_config_path, 'r') as f:
        base_sweep_config = yaml.safe_load(f)
    
    rides_config = load_rides_config(rides_config_path)
    
    # Create ride-specific sweep config
    sweep_config = create_sweep_config(base_sweep_config, ride_name, rides_config)
    
    # Add ride name to sweep name for clarity
    ride_display_name = rides_config['rides'][ride_name].get('display_name', ride_name)
    
    print(f"Setting up sweep for: {ride_display_name} ({ride_name})")
    print(f"Data path: {rides_config['rides'][ride_name]['data_path']}")
    
    # Create sweep
    sweep_id = wandb.sweep(sweep_config, project=project_name, entity=entity)
    return sweep_id

def main():
    parser = argparse.ArgumentParser(description="Run hyperparameter sweep for TCN models")
    parser.add_argument('--ride', required=True, help='Ride name (e.g., poseidon, atlantis)')
    parser.add_argument('--base-sweep-config', default='configs/sweep_config_base.yaml', 
                       help='Path to base sweep configuration file')
    parser.add_argument('--rides-config', default='configs/rides_config.yaml',
                       help='Path to rides configuration file')
    parser.add_argument('--count', type=int, default=50, help='Number of sweep runs')
    parser.add_argument('--wandb-project', default='waitless-tcn-hslu-dspro2-fs25',
                       help='Weights & Biases project name')
    parser.add_argument('--wandb-entity', default='waitless-hslu-dspro2-fs25',
                       help='Weights & Biases entity name')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show sweep config without running')
    
    args = parser.parse_args()
    
    try:
        # Setup sweep
        sweep_id = setup_sweep(
            args.base_sweep_config, 
            args.ride, 
            args.wandb_project, 
            args.wandb_entity,
            args.rides_config
        )
        
        if args.dry_run:
            print(f"Dry run complete. Sweep would be created with ID: {sweep_id}")
            return
        
        print(f"Starting sweep with ID: {sweep_id}")
        print(f"Running {args.count} experiments for ride: {args.ride}")
        
        # Run sweep
        wandb.agent(sweep_id, train_with_wandb, count=args.count)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    main()
