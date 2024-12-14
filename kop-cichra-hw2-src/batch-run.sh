#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <directory> <file> <numofruns> <config_path>"
    exit 1
fi

# Assign command-line arguments to variables
DIR=$1
FILE=$2
RUNS=$3
CONF=$4

# Check if the specified directory exists
if [ ! -d "$DIR" ]; then
    echo "Error: Directory $DIR does not exist."
    exit 1
fi

# Check if the specified file exists
if [ ! -f "$FILE" ]; then
    echo "Error: File $FILE does not exist."
    exit 1
fi

# Get absolute path of the file
ABS_FILE=$(realpath "$FILE")

# Loop through all *.mwcnf files in the specified directory
for mwcnf_file in "$DIR"/*.mwcnf; do
    if [ -f "$mwcnf_file" ]; then
        # Get absolute path of the current mwcnf file
        ABS_MWCNF=$(realpath "$mwcnf_file")
        
        echo "Processing file: $ABS_MWCNF with $ABS_FILE"
        
        # Run the command with absolute paths
        ./generate-fitness.sh "$ABS_MWCNF" "$ABS_FILE" "$RUNS" "$CONF"
    else
        echo "No *.mwcnf files found in $DIR."
    fi
done

echo "Script execution completed."
