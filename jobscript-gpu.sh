#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=8000

source .venv/bin/activate

cd lukasz_sawala_bsc_thesis/
python finetuningNN_maze.py


deactivate