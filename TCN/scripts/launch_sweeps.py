#!/usr/bin/env python3
"""
Launch multiple sweeps for different rides or configurations
"""

import argparse
import subprocess
import time
import yaml
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Launch multiple WandB sweeps")
    
    parser.add_argument('--rides', nargs='+', required=True,
                       help='List of ride names')
    parser.add_argument('--config', default='standard',
                       help='Sweep configuration to use')
    parser.add_argument('--count', type=int, default=50,
                       help='Number of runs per sweep')
    parser.add_argument('--parallel', type=int, default=1,
                       help='Number of parallel agents per sweep')
    parser.add_argument('--project', default='waitless-tcn-sweeps',
                       help='WandB project name')
    parser.add_argument('--entity', help='WandB entity')
    parser.add_argument('--delay', type=int, default=5,
                       help='Delay between launching sweeps (seconds)')
    
    return parser.parse_args()


def launch_sweep(ride_name, config, count, project, entity=None, parallel=1):
    """Launch a single sweep and return the process handles"""
    logger.info(f"Launching sweep for ride: {ride_name}")
    
    # First create the sweep
    cmd = [
        'python', 'sweep.py',
        '--ride', ride_name,
        '--config', config,
        '--project', project,
        '--count', '0'  # Don't run any agents yet
    ]
    
    if entity:
        cmd.extend(['--entity', entity])
    
    # Get sweep ID by running with count=0
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Extract sweep ID from output
    sweep_id = None
    for line in result.stdout.split('\n'):
        if 'Created new sweep:' in line:
            sweep_id = line.split('Created new sweep:')[1].strip()
            break
    
    if not sweep_id:
        logger.error(f"Failed to create sweep for {ride_name}")
        logger.error(result.stderr)
        return []
    
    logger.info(f"Created sweep {sweep_id} for {ride_name}")
    
    # Launch parallel agents
    processes = []
    runs_per_agent = count // parallel
    remaining_runs = count % parallel
    
    for i in range(parallel):
        agent_count = runs_per_agent
        if i == 0:
            agent_count += remaining_runs
        
        cmd = [
            'python', 'sweep.py',
            '--ride', ride_name,
            '--sweep-id', sweep_id,
            '--count', str(agent_count),
            '--project', project
        ]
        
        if entity:
            cmd.extend(['--entity', entity])
        
        logger.info(f"Launching agent {i+1}/{parallel} for {ride_name} (running {agent_count} runs)")
        
        process = subprocess.Popen(cmd)
        processes.append(process)
        time.sleep(2)  # Small delay between agents
    
    return processes


def main():
    args = parse_arguments()
    
    all_processes = []
    sweep_info = []
    
    # Launch sweeps for each ride
    for ride in args.rides:
        processes = launch_sweep(
            ride_name=ride,
            config=args.config,
            count=args.count,
            project=args.project,
            entity=args.entity,
            parallel=args.parallel
        )
        
        all_processes.extend(processes)
        sweep_info.append({
            'ride': ride,
            'processes': len(processes),
            'total_runs': args.count
        })
        
        if ride != args.rides[-1]:  # Don't delay after last ride
            logger.info(f"Waiting {args.delay} seconds before next sweep...")
            time.sleep(args.delay)
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("SWEEP SUMMARY")
    logger.info("="*50)
    for info in sweep_info:
        logger.info(f"Ride: {info['ride']}")
        logger.info(f"  - Parallel agents: {info['processes']}")
        logger.info(f"  - Total runs: {info['total_runs']}")
    logger.info(f"\nTotal processes running: {len(all_processes)}")
    logger.info("="*50)
    
    # Wait for all processes to complete
    logger.info("\nWaiting for all sweeps to complete...")
    logger.info("(Press Ctrl+C to stop all sweeps)")
    
    try:
        for i, process in enumerate(all_processes):
            process.wait()
            logger.info(f"Process {i+1}/{len(all_processes)} completed")
    except KeyboardInterrupt:
        logger.warning("\nStopping all sweeps...")
        for process in all_processes:
            process.terminate()
        logger.info("All sweeps terminated")
    
    logger.info("All sweeps completed!")


if __name__ == "__main__":
    main()