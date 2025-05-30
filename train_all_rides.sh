#!/bin/bash

# List of rides to train models for
rides=(
    "blue_fire_megacoaster"
    "poseidon"
    "silver_star"
    "arthur"
    "euromir"
    "alpine_express_enzian"
    "eurosat_cancan_coaster"
    "swiss_bob_run"
    "voletarium"
)

# Loop through each ride and train a model
for ride in "${rides[@]}"; do
    echo "===================="
    echo "Training model for: $ride"
    echo "===================="
    
    # Run the training command
    python -m TCN.train --ride "$ride"
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "Successfully trained model for $ride"
    else
        echo "Failed to train model for $ride"
        # Uncomment the next line if you want to stop on first failure
        # exit 1
    fi
    
    echo ""
done

echo "All training jobs completed!"
