#!/bin/bash
# Script to run tests in background and save output

source ~/miniconda3/etc/profile.d/conda.sh
conda activate env2.0

echo "Starting test at $(date)" > test_output.log

# Run the test in background with nohup
JAX_PLATFORM_NAME=cpu nohup python full_test.py >> test_output.log 2>&1 &

echo "Test running in background with PID: $!"
echo "Check progress with: tail -f test_output.log"