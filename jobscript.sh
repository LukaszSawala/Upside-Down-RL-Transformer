#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=4GB

source .venv/bin/activate
cd notebooks/
python k_means_distr.py

deactivate