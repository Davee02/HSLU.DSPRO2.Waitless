import os
import pickle
import logging
import torch
from pathlib import Path
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)


def save_models(gb_model, tcn_model, config: Dict, output_dir: str = "models/cached_scheduled_sampling"):
    """Save both GradientBoosting and TCN models"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    target_ride = config['target_ride'].replace(' ', '_')
    
    # Save GradientBoosting model
    gb_path = output_path / f"{target_ride}_gb_baseline.pkl"
    with open(gb_path, "wb") as f:
        pickle.dump(gb_model, f)
    
    # Save TCN model with config
    tcn_path = output_path / f"{target_ride}_cached_scheduled_sampling_tcn.pt"
    torch.save({
        'model_state_dict': tcn_model.state_dict(),
        'config': config,
        'model_config': tcn_model.get_config() if hasattr(tcn_model, 'get_config') else None,
        'cache_config': {
            'cache_update_frequency': config.get('cache_update_frequency', 5),
            'max_cache_size': config.get('max_cache_size', 100000),
            'sampling_strategy': config.get('sampling_strategy', 'linear'),
            'noise_factor': config.get('noise_factor', 0.15)
        }
    }, tcn_path)
    
    logger.info(f"Models saved: {gb_path}, {tcn_path}")
    
    return str(gb_path), str(tcn_path)


def load_models(ride_name: str, model_dir: str = "models/cached_scheduled_sampling", 
                device: Optional[torch.device] = None) -> Tuple[object, torch.nn.Module, Dict]:
    """Load trained models and configuration"""
    from models.tcn_model import AutoregressiveTCNModel
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_path = Path(model_dir)
    ride_name_normalized = ride_name.replace(' ', '_')
    
    # Load GradientBoosting model
    gb_path = model_path / f"{ride_name_normalized}_gb_baseline.pkl"
    with open(gb_path, "rb") as f:
        gb_model = pickle.load(f)
    
    # Load TCN model and config
    tcn_path = model_path / f"{ride_name_normalized}_cached_scheduled_sampling_tcn.pt"
    checkpoint = torch.load(tcn_path, map_location=device)
    
    config = checkpoint['config']
    
    # Create and load TCN model
    if 'model_config' in checkpoint and checkpoint['model_config']:
        tcn_model = AutoregressiveTCNModel.from_config(checkpoint['model_config'])
    else:
        # Fallback to creating from training config
        tcn_model = AutoregressiveTCNModel.from_config(config)
    
    tcn_model.load_state_dict(checkpoint['model_state_dict'])
    tcn_model.to(device)
    tcn_model.eval()
    
    logger.info(f"Loaded models for ride '{ride_name}' from {model_dir}")
    
    return gb_model, tcn_model, config


def create_model_from_config(config: Dict, device: Optional[torch.device] = None) -> torch.nn.Module:
    """Create a new TCN model from configuration"""
    from models.tcn_model import AutoregressiveTCNModel
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AutoregressiveTCNModel.from_config(config)
    model.to(device)
    
    # Enable torch.compile if specified
    if config.get('use_torch_compile', True) and hasattr(torch, 'compile'):
        logger.info("Enabling torch.compile for model")
        model = torch.compile(model, mode='reduce-overhead')
    
    return model