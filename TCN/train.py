#!/usr/bin/env python3
"""
Main training script for Autoregressive TCN with Cached Scheduled Sampling
"""

import argparse
import logging
import wandb
from pathlib import Path

from TCN.training.trainer import CachedScheduledSamplingTCNTrainer
from TCN.utils.config import load_config, create_config_from_ride
from TCN.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train autoregressive TCN with cached scheduled sampling"
    )
    
    # Configuration options
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--ride', help='Ride name (alternative to --config)')
    parser.add_argument('--rides-config', default='TCN/configs/rides_config.yaml',
                       help='Path to rides configuration file')
    
    # Training options
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    parser.add_argument('--seq-length', type=int, help='Sequence length')
    
    # Model options
    parser.add_argument('--num-channels', type=int, help='Number of TCN channels')
    parser.add_argument('--kernel-size', type=int, help='TCN kernel size')
    parser.add_argument('--num-layers', type=int, help='Number of TCN layers')
    parser.add_argument('--dropout', type=float, help='Dropout rate')
    
    # Cached scheduled sampling options
    parser.add_argument('--sampling-strategy', choices=['linear', 'exponential', 'inverse_sigmoid'],
                       help='Scheduled sampling strategy')
    parser.add_argument('--noise-factor', type=float, 
                       help='Noise factor for prediction uncertainty')
    parser.add_argument('--cache-update-frequency', type=int,
                       help='Update prediction cache every N epochs')
    parser.add_argument('--max-cache-size', type=int,
                       help='Maximum number of cached predictions')
    
    # Wandb options
    parser.add_argument('--wandb-project', default='waitless-autoregressive-tcn-hslu-dspro2-fs25',
                       help='Weights & Biases project name')
    parser.add_argument('--wandb-entity', default='waitless-hslu-dspro2-fs25',
                       help='Weights & Biases entity name')
    parser.add_argument('--wandb-run-name', help='Custom run name for wandb')
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')
    
    # Other options
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--log-file', help='Log file name')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--no-compile', action='store_true', 
                       help='Disable torch.compile')
    parser.add_argument('--no-mixed-precision', action='store_true',
                       help='Disable mixed precision training')
    
    return parser.parse_args()


def override_config_with_args(config, args):
    """Override configuration with command line arguments"""
    # Training options
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        config['learning_rate'] = args.learning_rate
    if args.seq_length is not None:
        config['seq_length'] = args.seq_length
    
    # Model options
    if args.num_channels is not None:
        config['num_channels'] = args.num_channels
    if args.kernel_size is not None:
        config['kernel_size'] = args.kernel_size
    if args.num_layers is not None:
        config['num_layers'] = args.num_layers
    if args.dropout is not None:
        config['dropout'] = args.dropout
    
    # Cached scheduled sampling options
    if args.sampling_strategy:
        config['sampling_strategy'] = args.sampling_strategy
    if args.noise_factor is not None:
        config['noise_factor'] = args.noise_factor
    if args.cache_update_frequency is not None:
        config['cache_update_frequency'] = args.cache_update_frequency
    if args.max_cache_size is not None:
        config['max_cache_size'] = args.max_cache_size
    
    # Other options
    if args.seed is not None:
        config['seed'] = args.seed
    if args.no_compile:
        config['use_torch_compile'] = False
    if args.no_mixed_precision:
        config['use_mixed_precision'] = False
    
    # Wandb options
    config['use_wandb'] = not args.no_wandb
    
    return config


def main():
    """Main training function"""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging(log_level=args.log_level, log_file=args.log_file)
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    elif args.ride:
        config = create_config_from_ride(args.ride, args.rides_config)
        logger.info(f"Created cached scheduled sampling configuration for ride: {args.ride}")
    else:
        raise ValueError("Either --config or --ride must be specified")
    
    # Override config with command line arguments
    config = override_config_with_args(config, args)
    
    # Log configuration
    logger.info("Training configuration:")
    for key, value in config.items():
        if key not in ['wandb_api_key']:  # Don't log sensitive info
            logger.info(f"  {key}: {value}")
    
    # Initialize wandb if enabled
    if config.get('use_wandb', True):
        # Determine run name
        if args.wandb_run_name:
            run_name = args.wandb_run_name
        else:
            run_name = f"{config['target_ride']}_cached_scheduled_sampling_tcn"
            if config.get('run_name'):
                run_name = f"{config['target_ride']}_{config['run_name']}"
        
        # Initialize wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=config,
            name=run_name,
            tags=[
                'cached_scheduled_sampling', 
                'autoregressive', 
                'tcn', 
                'gradientboosting', 
                config['target_ride'],
                config.get('sampling_strategy', 'linear')
            ]
        )
        logger.info(f"Initialized wandb run: {run_name}")
    
    try:
        # Create trainer and train model
        trainer = CachedScheduledSamplingTCNTrainer(config)
        metrics = trainer.train_model()
        
        logger.info("Training completed successfully!")
        logger.info(f"Final metrics: {metrics}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise
    
    finally:
        if wandb.run:
            wandb.finish()
    
    return 0


if __name__ == "__main__":
    exit(main())