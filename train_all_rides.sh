#!/bin/bash

rides=(
    "eurosat__cancan_coaster"
)

for ride in "${rides[@]}"; do
    echo "===================="
    echo "Training model for: $ride"
    echo "===================="

    python -m TCN.train --ride "$ride"

    if [ $? -eq 0 ]; then
        echo "Successfully trained model for $ride"
    else
        echo "Failed to train model for $ride"

    fi
    
    echo ""
done

echo "All training jobs completed!"
