#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=2GB

source .venv/bin/activate
cd lukasz_sawala_bsc_thesis/
python dataset_generation.py
deactivate