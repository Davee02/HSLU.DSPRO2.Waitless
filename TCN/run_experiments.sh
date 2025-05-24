
check_env() {
    echo "🔍 Environment Check:"
    echo "   Python: $(python --version 2>&1)"
    echo "   CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'PyTorch not installed')"
    echo "   Working Directory: $(pwd)"
    echo "   GPU Device: ${CUDA_VISIBLE_DEVICES}"
    
    if [ -f "configs/rides_config.yaml" ]; then
        echo "   Configured Rides: $(python -c "import yaml; print(len(yaml.safe_load(open('configs/rides_config.yaml')).get('rides', {})))" 2>/dev/null || echo 'unknown')"
    else
        echo "   ⚠️  No rides configured. Run 'make setup' first."
    fi
}#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}  

wandb_login() {
    echo "⚠️  Consider using: make login"
    wandb login
}

train_single() {
    local ride=$1
    echo "⚠️  Consider using: make train RIDE=$ride"
    python train.py --ride "$ride"
}

run_sweep() {
    local ride=$1
    local count=${2:-50}
    echo "⚠️  Consider using: make sweep RIDE=$ride COUNT=$count"
    python run_sweep.py --ride "$ride" --count "$count"
}

train_all_rides() {
    echo "⚠️  Consider using: make train-all"
    make train-all
}

sweep_all_rides() {
    local count=${1:-30}
    echo "⚠️  Consider using: make sweep-all COUNT=$count"
    make sweep-all COUNT="$count"
}

monitor_gpu() {
    echo "🖥️  GPU Monitoring (Press Ctrl+C to stop):"
    if command -v nvidia-smi &> /dev/null; then
        watch -n 2 nvidia-smi
    else
        echo "❌ nvidia-smi not found"
    fi
}

usage() {
    echo "TCN Training Helper Scripts"
    echo ""
    echo "⚠️  DEPRECATED: Most functionality moved to Makefile"
    echo "    Use 'make help' for current commands"
    echo ""
    echo "Remaining utilities:"
    echo "  $0 check-env        - Check environment setup"
    echo "  $0 monitor-gpu      - Monitor GPU usage"
    echo ""
    echo "Legacy commands (use 'make' instead):"
    echo "  $0 login           - Login to W&B (use: make login)"
    echo "  $0 single <ride>   - Train single model (use: make train RIDE=<ride>)"
    echo "  $0 sweep <ride>    - Run sweep (use: make sweep RIDE=<ride>)"
    echo "  $0 train-all       - Train all (use: make train-all)"
    echo "  $0 sweep-all       - Sweep all (use: make sweep-all)"
}

case "$1" in
    check-env)
        check_env
        ;;
    monitor-gpu)
        monitor_gpu
        ;;
    login)
        wandb_login
        ;;
    single)
        if [ -z "$2" ]; then
            echo "Error: Please specify ride name"
            echo "Better: make train RIDE=<ride>"
            exit 1
        fi
        train_single "$2"
        ;;
    sweep)
        if [ -z "$2" ]; then
            echo "Error: Please specify ride name"
            echo "Better: make sweep RIDE=<ride>"
            exit 1
        fi
        run_sweep "$2" "$3"
        ;;
    train-all)
        train_all_rides
        ;;
    sweep-all)
        sweep_all_rides "$2"
        ;;
    *)
        usage
        ;;
esac