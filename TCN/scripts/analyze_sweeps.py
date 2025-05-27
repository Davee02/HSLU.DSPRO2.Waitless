#!/usr/bin/env python3
"""
Analyze WandB sweep results and find best hyperparameters
"""

import argparse
import wandb
import pandas as pd
import numpy as np
from pathlib import Path
import json
import yaml


def parse_arguments():
    parser = argparse.ArgumentParser(description="Analyze WandB sweep results")
    
    parser.add_argument('--project', required=True, help='WandB project name')
    parser.add_argument('--entity', help='WandB entity')
    parser.add_argument('--sweep-id', help='Specific sweep ID to analyze')
    parser.add_argument('--ride', help='Filter by ride name')
    parser.add_argument('--metric', default='test_mae', help='Metric to optimize')
    parser.add_argument('--top-k', type=int, default=10, help='Number of top runs to show')
    parser.add_argument('--export', help='Export results to CSV file')
    parser.add_argument('--save-best-config', help='Save best config to YAML file')
    
    return parser.parse_args()


def get_sweep_runs(project, entity=None, sweep_id=None, ride=None):
    """Get all runs from a sweep or project"""
    api = wandb.Api()
    
    runs = []
    
    if sweep_id:
        # Get specific sweep
        sweep = api.sweep(f"{entity}/{project}/{sweep_id}" if entity else f"{project}/{sweep_id}")
        for run in sweep.runs:
            if ride and run.config.get('target_ride') != ride:
                continue
            runs.append(run)
    else:
        # Get all runs from project
        project_path = f"{entity}/{project}" if entity else project
        for run in api.runs(project_path):
            if run.sweep and (not ride or run.config.get('target_ride') == ride):
                runs.append(run)
    
    return runs


def analyze_runs(runs, metric='test_mae', top_k=10):
    """Analyze runs and find best hyperparameters"""
    data = []
    
    for run in runs:
        if run.state != 'finished':
            continue
        
        # Get config and metrics
        config = dict(run.config)
        summary = dict(run.summary)
        
        # Check if metric exists
        if metric not in summary:
            continue
        
        # Create row
        row = {
            'run_id': run.id,
            'run_name': run.name,
            'sweep_id': run.sweep.id if run.sweep else None,
            'ride': config.get('target_ride', 'unknown'),
            metric: summary[metric]
        }
        
        # Add important hyperparameters
        important_params = [
            'seq_length', 'batch_size', 'num_channels', 'kernel_size', 
            'num_layers', 'dropout', 'learning_rate', 'weight_decay',
            'sampling_strategy', 'noise_factor', 'cache_update_frequency',
            'gb_n_estimators', 'gb_max_depth', 'gb_learning_rate',
            'scheduler_type', 't_0', 't_mult', 'eta_min'
        ]
        
        for param in important_params:
            if param in config:
                row[param] = config[param]
        
        # Add other metrics
        other_metrics = ['test_rmse', 'test_r2', 'test_smape', 'test_mae_tf', 
                        'gap_mae_gap', 'gap_mae_gap_pct', 'val_mae']
        for m in other_metrics:
            if m in summary and m != metric:
                row[m] = summary[m]
        
        data.append(row)
    
    # Create dataframe and sort
    df = pd.DataFrame(data)
    df = df.sort_values(metric)
    
    return df


def print_analysis(df, metric, top_k):
    """Print analysis results"""
    print(f"\n{'='*80}")
    print(f"SWEEP ANALYSIS RESULTS")
    print(f"{'='*80}")
    print(f"Total runs analyzed: {len(df)}")
    print(f"Optimizing for: {metric}")
    print(f"\nTop {top_k} runs:")
    print(f"{'='*80}")
    
    # Show top runs
    top_df = df.head(top_k)
    
    for i, row in top_df.iterrows():
        print(f"\nRank {len(top_df) - len(top_df) + list(top_df.index).index(i) + 1}:")
        print(f"  Run: {row['run_name']} (ID: {row['run_id']})")
        print(f"  Ride: {row['ride']}")
        print(f"  {metric}: {row[metric]:.4f}")
        
        # Show other metrics if available
        if 'test_rmse' in row:
            print(f"  test_rmse: {row['test_rmse']:.4f}")
        if 'test_r2' in row:
            print(f"  test_r2: {row['test_r2']:.4f}")
        if 'test_smape' in row:
            print(f"  test_smape: {row['test_smape']:.2f}%")
        if 'gap_mae_gap_pct' in row:
            print(f"  AR vs TF gap: {row['gap_mae_gap_pct']:.1f}%")
        
        print(f"\n  Hyperparameters:")
        print(f"    Architecture: channels={row.get('num_channels')}, "
              f"layers={row.get('num_layers')}, kernel={row.get('kernel_size')}")
        print(f"    Sequence: length={row.get('seq_length')}, batch={row.get('batch_size')}")
        print(f"    Training: lr={row.get('learning_rate'):.5f}, "
              f"dropout={row.get('dropout'):.2f}, wd={row.get('weight_decay'):.5f}")
        print(f"    Sampling: strategy={row.get('sampling_strategy')}, "
              f"noise={row.get('noise_factor'):.2f}, cache_freq={row.get('cache_update_frequency')}")
        print(f"    GB: estimators={row.get('gb_n_estimators')}, "
              f"depth={row.get('gb_max_depth')}, lr={row.get('gb_learning_rate'):.3f}")
    
    print(f"\n{'='*80}")
    
    # Aggregate statistics by ride
    if 'ride' in df.columns:
        print("\nPer-ride statistics:")
        print(f"{'='*80}")
        ride_stats = df.groupby('ride')[metric].agg(['count', 'min', 'mean', 'std'])
        print(ride_stats)
    
    # Parameter importance (if enough runs)
    if len(df) >= 20:
        print(f"\n{'='*80}")
        print("Parameter correlations with performance:")
        print(f"{'='*80}")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        param_cols = [col for col in numeric_cols if col not in 
                     ['test_mae', 'test_rmse', 'test_r2', 'test_smape', 
                      'test_mae_tf', 'val_mae', 'gap_mae_gap', 'gap_mae_gap_pct']]
        
        correlations = []
        for col in param_cols:
            if df[col].nunique() > 1:  # Only parameters that vary
                corr = df[col].corr(df[metric])
                if not np.isnan(corr):
                    correlations.append((col, corr))
        
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        for param, corr in correlations[:10]:
            direction = "↓" if corr < 0 else "↑"
            print(f"  {param}: {corr:+.3f} {direction}")


def save_best_config(df, metric, output_path):
    """Save the best configuration to a YAML file"""
    best_run = df.iloc[0]
    
    # Extract config
    config = {
        'ride': best_run['ride'],
        'best_run_id': best_run['run_id'],
        'best_run_name': best_run['run_name'],
        f'best_{metric}': float(best_run[metric]),
        'hyperparameters': {}
    }
    
    # Add hyperparameters
    param_cols = [col for col in df.columns if col not in 
                 ['run_id', 'run_name', 'sweep_id', 'ride'] and 
                 not col.startswith('test_') and not col.startswith('val_') and
                 not col.startswith('gap_')]
    
    for param in param_cols:
        if pd.notna(best_run[param]):
            value = best_run[param]
            # Convert numpy types to Python types
            if isinstance(value, np.integer):
                value = int(value)
            elif isinstance(value, np.floating):
                value = float(value)
            config['hyperparameters'][param] = value
    
    # Save to file
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\nBest configuration saved to: {output_path}")


def main():
    args = parse_arguments()
    
    print("Fetching runs from WandB...")
    runs = get_sweep_runs(
        project=args.project,
        entity=args.entity,
        sweep_id=args.sweep_id,
        ride=args.ride
    )
    
    if not runs:
        print("No runs found!")
        return
    
    print(f"Found {len(runs)} runs")
    
    # Analyze runs
    df = analyze_runs(runs, metric=args.metric, top_k=args.top_k)
    
    if df.empty:
        print(f"No completed runs found with metric '{args.metric}'")
        return
    
    # Print analysis
    print_analysis(df, args.metric, args.top_k)
    
    # Export if requested
    if args.export:
        df.to_csv(args.export, index=False)
        print(f"\nResults exported to: {args.export}")
    
    # Save best config if requested
    if args.save_best_config:
        save_best_config(df, args.metric, args.save_best_config)


if __name__ == "__main__":
    main()