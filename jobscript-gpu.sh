#!/bin/bash
#SBATCH --time=16:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=8000

source .venv/bin/activate

cd lukasz_sawala_bsc_thesis/
python ft-selfimprove-rollout-lastcondition.py


deactivate