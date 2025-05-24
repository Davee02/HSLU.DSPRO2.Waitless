# TCN Wait Time Prediction - CLI Training Suite

A complete command-line interface for training Temporal Convolutional Networks (TCN) for wait time prediction across different rides. Features smart initialization, centralized configuration management, and a clean tool architecture.

## 🚀 Quick Start

### 1. Setup

```bash
# Initialize everything
python setup.py

# Login to Weights & Biases
make login
```

### 2. Add Your Rides

```bash
# Add rides to the central configuration
make add-ride RIDE=poseidon PATH="../data/processed/ep/rides/poseidon.parquet"

# Check what you've added
make list-rides
```

### 3. Start Training

```bash
# Train a single model
make train RIDE=poseidon

# Run hyperparameter sweep (50 runs)
make sweep RIDE=poseidon COUNT=50

# Train all configured rides
make train-all

# Run sweeps for all rides (30 runs each)
make sweep-all COUNT=30
```

## 🏗️ Architecture & Tools

This project uses three complementary tools, each with a clear purpose:

| Tool | Purpose | When to Use | Example |
|------|---------|-------------|---------|
| **setup.py** | 🚀 Smart initialization | First-time setup | `python setup.py` |
| **Makefile** | 🎯 Daily workflow | All training operations | `make train RIDE=poseidon` |
| **run_experiments.sh** | 🔧 Environment utilities | Environment checks, GPU monitoring | `./run_experiments.sh check-env` |

## 📁 Project Structure

```
├── train.py                         # Main training script
├── run_sweep.py                    # Hyperparameter sweep runner
├── setup.py                       # 🆕 Smart initialization script
├── Makefile                        # 🎯 Main command interface
├── run_experiments.sh              # 🔧 Environment utilities
├── requirements.txt                # Python dependencies
├── configs/                        # 🆕 Configuration directory
│   ├── rides_config.yaml          #     All ride configurations
│   ├── sweep_config_base.yaml     #     Base sweep config for all rides
│   └── config_poseidon_example.yaml #   Example individual config
├── models/                         # Saved models
├── logs/                          # Training logs
└── README.md                      # This guide
```

## ⚙️ Configuration System

### **🆕 Centralized Configuration Approach**

The system uses centralized configuration management to eliminate duplication and simplify ride management:

#### **configs/rides_config.yaml** - All ride-specific settings
```yaml
rides:
  poseidon:
    data_path: "../data/processed/ep/rides/poseidon.parquet"
    display_name: "Poseidon"
    description: "Water ride with high capacity"

global_settings:
  splits_output_dir: "../data/processed/splits"
  model_output_dir: "./models"

default_params:
  epochs: 100
  patience: 10
  seed: 42
  use_wandb: true
```

#### **configs/sweep_config_base.yaml** - Base hyperparameter ranges
```yaml
method: bayes
metric:
  name: combined_mae
  goal: minimize

parameters:
  # Ride-specific params (data_path, target_ride) injected dynamically
  seq_length:
    values: [24, 48, 96, 192, 384]
  batch_size:
    values: [128, 256, 512, 1024]
  num_channels:
    values: [32, 64, 128, 256]
  kernel_size:
    values: [2, 3, 5, 8]
  dropout:
    values: [0.1, 0.2, 0.3]
  learning_rate:
    values: [1e-4, 10**-4.5, 1e-5, 10**-5.5, 1e-6]
```

## 🎯 Command Reference

### **Primary Interface **

```bash
# Setup and initialization
make help                           # Show all commands
python setup.py                     # Smart initial setup
make login                          # Login to W&B

# Ride management
make add-ride RIDE=newride PATH=/path/to/data.parquet
make list-rides                     # Show configured rides

# Training operations
make train RIDE=poseidon            # Train single model
make sweep RIDE=poseidon COUNT=100  # Run hyperparameter sweep
make train-all                      # Train all configured rides
make sweep-all COUNT=50             # Sweep all rides

# Utilities
make clean                          # Clean up generated files
```

### **Direct Python Usage**

```bash
# Train single model
python train.py --ride poseidon
python train.py --config configs/custom_config.yaml  # Alternative

# Run sweep
python run_sweep.py --ride poseidon --count 100
python run_sweep.py --ride atlantis --dry-run        # Preview config

# With custom W&B settings
python train.py --ride poseidon \
                --wandb-project my-project \
                --wandb-entity my-entity
```

### **Environment Utilities**

```bash
# Environment diagnostics
./run_experiments.sh check-env      # Check Python, CUDA, configs
./run_experiments.sh monitor-gpu    # Real-time GPU monitoring
```

## 📊 Experiment Tracking & Monitoring

### **Weights & Biases Integration**

All experiments automatically track:
- **Training/validation losses** with learning rate schedules
- **Final evaluation metrics** (MAE, RMSE, R²)
- **Model artifacts** (linear + TCN models)
- **System metrics** (GPU usage, memory)
- **Hyperparameter combinations** for sweeps

### **Key Metrics Tracked**
- `combined_mae`: Combined model MAE (primary optimization metric)
- `combined_rmse`: Combined model RMSE
- `combined_r2`: Combined model R²
- `linear_mae`: Linear baseline MAE
- `best_val_loss`: Best validation loss during training

### **Real-time Monitoring**

Training progress is automatically logged to:
- **W&B Dashboard** - Real-time metrics and visualizations
- **Console Output** - Live training logs during execution
- **Log Files** - `training.log` for detailed logs

```bash
# Check GPU usage during training
./run_experiments.sh monitor-gpu

# View recent console output
tail -f training.log
```

## 🎯 Advanced Usage

### **GPU Management**
```bash
# Use specific GPU
export CUDA_VISIBLE_DEVICES=0
make train RIDE=poseidon

# Check current GPU setup
./run_experiments.sh check-env
```

### **Preview Mode**
```bash
# Preview sweep configuration without running
python run_sweep.py --ride poseidon --dry-run
```

### **Custom Configurations**
```bash
# Use different base sweep config
python run_sweep.py --ride poseidon \
                    --base-sweep-config configs/sweep_config_aggressive.yaml

# Use custom rides config
python train.py --ride poseidon \
                --rides-config configs/custom_rides.yaml
```


## 📈 Scaling to Multiple Rides

### **Adding New Rides**
```bash
# Method 1: Quick add (recommended)
make add-ride RIDE=hurricane PATH="../data/processed/ep/rides/hurricane.parquet"

# Method 2: Manual edit
# Edit configs/rides_config.yaml directly
```
