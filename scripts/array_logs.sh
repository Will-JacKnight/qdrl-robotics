#!/bin/bash
#PBS -lwalltime=01:00:00
#PBS -lselect=1:ncpus=1:mem=16gb:ngpus=1:gpu_type=L40S

# Get the base directory (assuming script is run from project root)
BASE_DIR="/rds/general/user/jw1524/home/qdrl-robotics"
OUTPUT_BASE="outputs/final"

# Change to the project directory
cd "$BASE_DIR"

# Find all subdirectories under outputs/final
for output_dir in "$OUTPUT_BASE"/*/; do
    # Remove trailing slash and get just the directory name
    output_path="${output_dir%/}"
    
    # Skip if it's not a directory or if it's the logs directory
    if [[ ! -d "$output_path" ]] || [[ "$(basename "$output_path")" == "logs" ]]; then
        continue
    fi
    
    echo "Processing: $output_path"
    
    # Run the Python command for this output path
    python main.py --output_path "$output_path"
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "Successfully processed: $output_path"
    else
        echo "Error processing: $output_path"
    fi
    
    echo "----------------------------------------"
done

echo "All directories processed."
