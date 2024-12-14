#!/bin/bash

# Check required parameters
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <problem_file> <optima_file> <numofruns> <config_file>"
    exit 1
fi

PROBLEM=$1       # The .mwcnf file representing the problem
OPTIMA=$2        # The file with optimas
RUNS=$3          # Number of batch runs for simulated annealing
CONFIG=$4        # JSON file with parameters
# Extract file names
PROBLEM_BASENAME=$(basename "$PROBLEM")           # e.g., problem.mwcnf
PROBLEM_NAME="${PROBLEM_BASENAME%.*}"             # e.g., problem (without extension)
OPTIMA_BASENAME=$(basename "$OPTIMA")             # e.g., optima.txt

# Generate unique ID for the test
UNIQUE_ID=$(date +%s)_$(uuidgen)

# Create working directory
WORKING_DIR="fitness_test_${PROBLEM_NAME}_${UNIQUE_ID}"
mkdir -p "$WORKING_DIR/inputs" "$WORKING_DIR/outputs"
echo "Created working directory: $WORKING_DIR"

# Copy input files
cp "$PROBLEM" "$WORKING_DIR/inputs/"
cp "$OPTIMA" "$WORKING_DIR/inputs/"
cp "$CONFIG" "$WORKING_DIR/inputs/"
echo "Copied problem and optima files to inputs directory."

# Change to the working directory
cd "$WORKING_DIR" || exit

# Run batch-anneal.py
sleep 1
echo "Running batch-anneal.py..."
python3 scripts/batch-anneal.py \
    --data_dir "inputs" \
    --optima_file "inputs/$OPTIMA_BASENAME" \
    --out_dir "." \
    --batch_runs "$RUNS" \
    --config "$CONFIG"

# Run plot-avgs.py
echo "Generating plots with plot-avgs.py..."
python3 scripts/plot-avgs.py \
    --data_dir "." \
    --filename "$PROBLEM_NAME" \
    --out_dir "outputs"

#cd $WORKING_DIR
rm -rf *.csv #remove to keep detailed logs of every run (takes a lot of space)
# Completion message
echo "Process complete. Outputs are in: $WORKING_DIR/outputs"
