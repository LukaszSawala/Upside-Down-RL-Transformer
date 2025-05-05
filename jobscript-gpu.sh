#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=8000

source .venv/bin/activate

cd lukasz_sawala_bsc_thesis/
python model_evaluation.py --episodes 10 --model_type DecisionTransformer --d_r_array_length 35


deactivate