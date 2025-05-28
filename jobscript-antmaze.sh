#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=2GB

source .venv/bin/activate
cd lukasz_sawala_bsc_thesis/
python transfer_eval-main.py --episodes 10 --model_type ANTMAZE_BERT_MLP --d_r_array_length 21
deactivate