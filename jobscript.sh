#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=4GB

source .venv/bin/activate
cd lukasz_sawala_bsc_thesis/
python model_evaluation.py --episodes 1

deactivate