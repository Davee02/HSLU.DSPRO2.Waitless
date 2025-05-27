#!/usr/bin/env python3
"""
WandB Sweep Runner for Autoregressive TCN with Cached Scheduled Sampling
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import logging
import wandb
import yaml
from pathlib import Path
from typing import Dict, Optional
import uuid

from configs.sweep_configs import get_sweep_config, list_sweep_configs, create_custom_sweep_config
from training.trainer import CachedScheduledSamplingTCNTrainer
from utils.config import create_config_from_ride
from utils.logging import setup_logging

logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments for sweep"""
    parser = argparse.ArgumentParser(
        description="Run WandB sweeps for autoregressive TCN training"
    )
    
    # Required arguments
    parser.add_argument('--ride', required=True, help='Ride name for training')
    
    # Sweep configuration
    parser.add_argument('--config', default='standard',
                       help=f'Sweep configuration name. Available: {list_sweep_configs()}')
    parser.add_argument('--sweep-id', help='Existing sweep ID to continue')
    parser.add_argument('--count', type=int, default=1,
                       help='Number of sweep runs to execute')
    
    # Project settings
    parser.add_argument('--project', default='waitless-tcn-hslu-dspro2-fs25',
                       help='WandB project name')
    parser.add_argument('--entity', default='waitless-hslu-dspro2-fs25', help='WandB entity name')
    parser.add_argument('--sweep-name', help='Custom name for the sweep')
    
    # Data paths
    parser.add_argument('--data-path', help='Override data path')
    parser.add_argument('--splits-dir', help='Override splits directory')
    
    # Additional options
    parser.add_argument('--dry-run', action='store_true',
                       help='Print sweep config without running')
    parser.add_argument('--save-config', help='Save sweep config to file')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    # Override specific parameters
    parser.add_argument('--override', nargs=2, metavar=('KEY', 'VALUE'), 
                       action='append', help='Override sweep parameters')
    
    return parser.parse_args()


def create_sweep_config_for_ride(ride_name: str, sweep_template: str, 
                                overrides: Optional[list] = None) -> Dict:
    """Create a complete sweep configuration for a specific ride"""
    # Get base sweep config
    sweep_config = get_sweep_config(sweep_template)
    
    # Apply any overrides
    if overrides:
        for key, value in overrides:
            # Try to parse value as number or boolean
            try:
                value = eval(value)
            except:
                pass  # Keep as string
            
            if '.' in key:
                # Handle nested keys like 'metric.name'
                parts = key.split('.')
                current = sweep_config
                for part in parts[:-1]:
                    current = current[part]
                current[parts[-1]] = value
            else:
                # Direct parameter override
                if key in sweep_config['parameters']:
                    if isinstance(value, list):
                        sweep_config['parameters'][key] = {'values': value}
                    else:
                        sweep_config['parameters'][key] = {'value': value}
                else:
                    # Add new parameter
                    sweep_config['parameters'][key] = {'value': value}
    
    # Add ride-specific parameters
    sweep_config['parameters']['target_ride'] = {'value': ride_name}
    
    # Set default name if not specified
    if 'name' not in sweep_config:
        sweep_config['name'] = f"{ride_name}_{sweep_template}_sweep"
    
    return sweep_config


def train_with_sweep_config():
    """Training function to be called by wandb agent"""
    # Initialize wandb run (wandb agent will have set the config)
    run = wandb.init()
    
    # Get the sweep parameters
    sweep_config = wandb.config
    
    # Create base configuration for the ride
    base_config = create_config_from_ride(
        sweep_config.target_ride,
        sweep_config.get('rides_config_path', 'TCN/configs/rides_config.yaml')
    )
    
    # Override with sweep parameters
    for key, value in sweep_config.items():
        base_config[key] = value
    
    # Handle scheduler configuration
    scheduler_type = base_config.get('scheduler_type', 'OneCycle')
    if scheduler_type == 'CosineAnnealingWarmRestarts':
        # Make sure required parameters are present
        base_config['scheduler_config'] = {
            'type': scheduler_type,
            't_0': base_config.get('t_0', 15),
            't_mult': base_config.get('t_mult', 1),
            'eta_min': base_config.get('eta_min', 1e-6)
        }
    elif scheduler_type == 'StepLR':
        base_config['scheduler_config'] = {
            'type': scheduler_type,
            'step_size': base_config.get('step_size', 20),
            'gamma': base_config.get('gamma', 0.5)
        }
    elif scheduler_type == 'CosineAnnealingLR':
        base_config['scheduler_config'] = {
            'type': scheduler_type,
            'T_max': base_config.get('epochs', 150),
            'eta_min': base_config.get('eta_min', 1e-6)
        }
    else:  # OneCycle
        base_config['scheduler_config'] = {
            'type': 'OneCycle',
            'pct_start': base_config.get('pct_start', 0.1)
        }
    
    # Log configuration
    logger.info("Training with sweep configuration:")
    for key, value in base_config.items():
        if key not in ['wandb_api_key']:
            logger.info(f"  {key}: {value}")
    
    try:
        
        print()
        trainer = CachedScheduledSamplingTCNTrainer(base_config)
        metrics = trainer.train_model()
        
        model_path = base_config.get('best_model_path', f'TCN/models/{base_config.target_ride}_cached_scheduled_sampling_tcn.pt')

        # Log the best model to WandB
        if os.path.exists(model_path):
            artifact = wandb.Artifact(
                name=f"best_model_{run.id}",
                type="model",
                description="Best performing model during sweep run",
                metadata=metrics  
            )
            artifact.add_file(model_path)
            run.log_artifact(artifact)
            logger.info(f"Best model logged to WandB as artifact: {artifact.name}")
        else:
            logger.warning(f"Model file not found at: {model_path}, skipping artifact upload.")

        logger.info(f"Training completed. Final metrics: {metrics}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        # Mark run as failed
        wandb.finish(exit_code=1)
        raise
    
    # Finish run
    wandb.finish()


def main():
    """Main function for running sweeps"""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(log_level=args.log_level)
    
    # Create sweep configuration
    logger.info(f"Creating sweep configuration for ride: {args.ride}")
    sweep_config = create_sweep_config_for_ride(
        args.ride, 
        args.config,
        args.override
    )
    
    # Override data paths if specified
    if args.data_path:
        sweep_config['parameters']['data_path'] = {'value': args.data_path}
    if args.splits_dir:
        sweep_config['parameters']['splits_output_dir'] = {'value': args.splits_dir}
    
    # Set sweep name
    if args.sweep_name:
        unique_id = uuid.uuid4().hex[:6]  # Short random suffix
        sweep_config['name'] = f"{args.ride}_{args.sweep_name}_{unique_id}"
    
    # Save config if requested
    if args.save_config:
        with open(args.save_config, 'w') as f:
            yaml.dump(sweep_config, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Sweep configuration saved to: {args.save_config}")
    
    # Dry run - just print config
    if args.dry_run:
        print("\nSweep Configuration:")
        print(yaml.dump(sweep_config, default_flow_style=False, sort_keys=False))
        return
    
    # Initialize or continue sweep
    if args.sweep_id:
        # Continue existing sweep
        sweep_id = args.sweep_id
        logger.info(f"Continuing existing sweep: {sweep_id}")
    else:
        # Create new sweep
        sweep_id = wandb.sweep(
            sweep_config,
            project=args.project,
            entity=args.entity
        )
        logger.info(f"Created new sweep: {sweep_id}")
        logger.info(f"View sweep at: https://wandb.ai/{args.entity or 'your-entity'}/{args.project}/sweeps/{sweep_id}")
    
    # Run sweep agent
    logger.info(f"Starting {args.count} sweep runs...")
    wandb.agent(
        sweep_id,
        function=train_with_sweep_config,
        count=args.count,
        project=args.project,
        entity=args.entity
    )
    
    logger.info("Sweep completed!")


if __name__ == "__main__":
    main()