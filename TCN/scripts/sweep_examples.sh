#!/bin/bash
# Example commands for running sweeps

# 1. Run a standard sweep for a single ride
python sweep.py --ride "blue_fire_megacoaster" --config standard --count 50

# 2. Run a quick sweep for testing
python sweep.py --ride "blue_fire_megacoaster" --config quick --count 10

# 3. Run a comprehensive sweep with custom name
python sweep.py --ride "blue_fire_megacoaster" \
    --config comprehensive \
    --count 100 \
    --sweep-name "bluefire_comprehensive_search" \
    --entity your-wandb-entity

# 4. Continue an existing sweep
python sweep.py --ride "blue_fire_megacoaster" \
    --sweep-id "abc123xyz" \
    --count 20

# 5. Run sweep with parameter overrides
python sweep.py --ride "blue_fire_megacoaster" \
    --config standard \
    --override epochs 200 \
    --override batch_size "[64, 128, 256]" \
    --override learning_rate "[0.01, 0.001, 0.0001]" \
    --count 30

# 6. Dry run to see sweep configuration
python sweep.py --ride "blue_fire_megacoaster" \
    --config standard \
    --dry-run

# 7. Save sweep configuration for later use
python sweep.py --ride "blue_fire_megacoaster" \
    --config standard \
    --save-config "configs/my_sweep_config.yaml" \
    --dry-run

# 8. Launch multiple sweeps for different rides
python scripts/launch_sweeps.py \
    --rides "blue_fire_megacoaster" "Space Mountain" "Thunder Mountain" \
    --config standard \
    --count 50 \
    --parallel 2

# 9. Analyze sweep results
python scripts/analyze_sweeps.py \
    --project waitless-tcn-sweeps \
    --entity your-wandb-entity \
    --metric test_mae \
    --top-k 20 \
    --export results/sweep_analysis.csv \
    --save-best-config configs/best_hyperparameters.yaml

# 10. Analyze specific sweep
python scripts/analyze_sweeps.py \
    --project waitless-tcn-sweeps \
    --sweep-id "abc123xyz" \
    --ride "blue_fire_megacoaster" \
    --metric test_mae \
    --top-k 5

# 11. Run scheduler-focused sweep
python sweep.py --ride "blue_fire_megacoaster" \
    --config scheduler_focused \
    --count 40 \
    --sweep-name "scheduler_optimization"

# 12. Create and run custom sweep from YAML
cat > custom_sweep.yaml << EOF
method: random
metric:
  name: test_mae
  goal: minimize
parameters:
  target_ride:
    value: blue_fire_megacoaster
  epochs:
    value: 100
  seq_length:
    distribution: int_uniform
    min: 48
    max: 384
  learning_rate:
    distribution: log_uniform_values
    min: 0.0001
    max: 0.01
  num_channels:
    values: [128, 256, 512]
EOF

wandb sweep custom_sweep.yaml --project waitless-tcn-sweeps
# Then use the sweep ID to run agents