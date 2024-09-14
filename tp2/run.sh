#!/bin/bash

start=1
end=10

for i in $(seq $start $end)
do
    pipenv run python main.py config/selection/boltzmann/2.t0/boltzmann-t0-${i}0.json
    echo "------------------------"
    
done

echo "All scripts have been executed."