#!/bin/bash

# Set SLURM / hardware environment
#SBATCH --job-name=convert_model
#SBATCH --output=logs/convert_model.out
#SBATCH --error=logs/convert_model_error.out
#SBATCH --account=def-pasquier
#SBATCH --nodes=1            # total nb of nodes
#SBATCH --cpus-per-task=32   # nb of CPU cores per task
#SBATCH --mem=100G
#SBATCH --time=3:00:00

# Default values
output_dir=""

# Parse options
while getopts ":hn:m:o:" option; do
   case $option in
      h) # display Help
         echo "Usage: $0 -n checkpoint_number -m model_name [-o output_folder]"
         echo "Options:"
         echo "  -h                Display help"
         echo "  -n <checkpoint>   Set checkpoint number (required)"
         echo "  -m <model>        Set model name (required)"
         echo "  -o <folder>       Optional output folder path"
         exit;;
      n) ckpt=$OPTARG;;
      m) model=$OPTARG;;
      o) output_dir=$OPTARG;;
      \?) echo "Error: Invalid option -$OPTARG" >&2; exit 1;;
      :) echo "Error: Option -$OPTARG requires an argument." >&2; exit 1;;
   esac
done

# Check required arguments
if [ -z "$ckpt" ] || [ -z "$model" ]; then
    echo "Error: -n and -m options are required."
    exit 1
fi

# Handle output folder if provided
if [ -n "$output_dir" ]; then
    # Basic syntax validation (no null bytes or control characters)
    if [[ "$output_dir" =~ [[:cntrl:]] ]]; then
        echo "Error: Output folder path contains invalid characters."
        exit 3
    fi

    # Create it if it doesn't exist
    if [ ! -d "$output_dir" ]; then
        echo "Creating output directory: $output_dir"
        mkdir -p "$output_dir" || {
            echo "Error: Failed to create directory $output_dir"
            exit 4
        }
    fi
fi

cd "runs/$model/checkpoint-$ckpt" || {
    echo "Error: Failed to change directory to runs/$model/checkpoint-$ckpt"
    exit 5
}

module purge
module load gcc arrow/17.0.0 cuda/12.2
source ../../../.venv/bin/activate

# Set output path for the model file
if [ -n "$output_dir" ]; then
    output_path="$output_dir/pytorch_model.bin"
    cp ./config.json ../../../$output_dir/
else
    output_path="pytorch_model.bin"
fi

# Run the Python script
python3 zero_to_fp32.py . "../../../$output_path" -d -t "global_step$ckpt"
status=$?

if [ $status -ne 0 ]; then
    echo "Error: zero_to_fp32.py failed with exit code $status"
    exit 6
fi

echo "Conversion completed successfully. Output: $output_path"