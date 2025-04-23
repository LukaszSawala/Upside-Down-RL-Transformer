#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=2GB

source .venv/bin/activate
cd lukasz_sawala_bsc_thesis/
python model_evaluation_c2.py --episodes 10 --model_type DecisionTransformer --d_r_array_length 35
deactivate