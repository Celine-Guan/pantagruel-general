#!/bin/bash

ENV_NAME=Pantagrueltest  # can change to other name
ENV_FILE=environment.yml
REQ_FILE=requirements.txt

echo "Creating conda environment from $ENV_FILE..."
conda env create -f $ENV_FILE -n $ENV_NAME

echo "Activating conda environment: $ENV_NAME"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

echo "Installing pip packages from $REQ_FILE..."
pip install -r $REQ_FILE

echo "Installing custom transformers branch..."
pip install git+https://github.com/formiel/transformers.git@pantagruel

echo "Setup complete! To activate the environment later, run:"
echo "conda activate $ENV_NAME"
