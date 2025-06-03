#!/usr/bin/env python3
"""
Setup script for TCN training environment
Creates necessary directories and default configuration files
"""

import os
import yaml
from pathlib import Path

def create_directory_structure():
    """Create necessary directories"""
    directories = [
        "configs",
        "models", 
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}/")

def create_default_rides_config():
    """Create default rides configuration file"""
    config_path = "configs/rides_config.yaml"
    
    if os.path.exists(config_path):
        print(f"⚠ {config_path} already exists, skipping...")
        return
    
    default_config = {
        'rides': {
            'poseidon': {
                'data_path': "../data/processed/ep/rides/poseidon.parquet",
                'display_name': "Poseidon",
                'description': "Water ride with high capacity"
            }
        },
        'global_settings': {
            'splits_output_dir': "../data/processed/splits",
            'base_data_dir': "../data/processed/ep/rides",
            'model_output_dir': "./models"
        },
        'default_params': {
            'epochs': 100,
            'patience': 10,
            'seed': 42,
            'use_wandb': True
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False, indent=2)
    
    print(f"✓ Created default rides config: {config_path}")


def create_gitignore():
    """Create .gitignore for the project"""
    gitignore_path = ".gitignore"
    
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
env.bak/
venv.bak/

# Jupyter Notebook
.ipynb_checkpoints

# Models and outputs
models/*.pt
models/*.pkl
logs/*.log

# Weights & Biases
wandb/

# Data files (uncomment if you don't want to track data)
# data/
# *.parquet
# *.csv

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
"""
    
    if not os.path.exists(gitignore_path):
        with open(gitignore_path, 'w') as f:
            f.write(gitignore_content)
        print(f"✓ Created .gitignore")
    else:
        print(f"⚠ .gitignore already exists, skipping...")

def print_next_steps():
    """Print helpful next steps for the user"""
    print("\n" + "="*60)
    print(" Setup Complete!")
    print("="*60)
    print("\n Directory structure created:")
    print("   configs/     - Configuration files")
    print("   models/      - Saved models")
    print("   logs/        - Training logs")
    
    print("\n Next steps:")
    print("   1. Login to Weights & Biases:")
    print("      make login")
    print("   ")
    print("   2. Add your rides:")
    print("      make add-ride RIDE=yourride PATH=/path/to/data.parquet")
    print("      # or edit configs/rides_config.yaml manually")
    print("   ")
    print("   3. Start training:")
    print("      make train RIDE=poseidon")
    print("      make sweep RIDE=poseidon COUNT=50")
    print("   ")
    print("   4. Batch operations:")
    print("      make train-all")
    print("      make sweep-all COUNT=30")
    
    print("\n Available commands:")
    print("   make help           - Show all commands")
    print("   make list-rides     - Show configured rides")
    print("   ./run_experiments.sh check-env  - Check environment")
    
    print("\n Configuration files:")
    print("   configs/rides_config.yaml      - Add/edit rides here")
    print("   configs/sweep_config_base.yaml - Modify hyperparameters here")
    
    print("\n Documentation:")
    print("   README.md - Complete usage guide and examples")

def main():
    print("Setting up TCN training environment...")
    print("-" * 40)
    
    create_directory_structure()
    create_default_rides_config()
    create_gitignore()
    
    print_next_steps()

if __name__ == "__main__":
    main()