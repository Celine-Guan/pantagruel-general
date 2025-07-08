#!/bin/bash

# Set environment name (must match the one in environment.yml)
ENV_NAME="sequence-classification"

echo ">>> [1/4] Checking if Conda is installed..."
if ! command -v conda &> /dev/null
then
    echo "Conda is not installed. Please install Miniconda or Anaconda first."
    exit 1
fi

echo ">>> [2/4] Creating Conda environment (if not already exists)..."
if conda info --envs | grep -q "^$ENV_NAME"; then
    echo "Environment '$ENV_NAME' already exists. Skipping creation."
else
    conda env create -f environment.yml
fi

echo ">>> [3/4] Activating environment and installing dependencies..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# pip dependencies are included in environment.yml
echo "All dependencies are installed (environment: $ENV_NAME)."

echo ">>> [4/4] Setup complete. Active environment: $ENV_NAME"
echo "You can now run training scripts, e.g.:"
echo "    python main.py"
