import argparse
import uuid
import torch
import numpy as np
import random


def set_seed(seed: int = 0) -> None:
    """
    Sets seeds for numpy, random, torch and torch.cuda modules.

    Parameters:
    seed (int): The seed value. Defaults to 0.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def parse_arguments(training: bool = False) -> dict:
    """
    Function defining and returning command line arguments depending on the desired mode (training/evaluation).
    """
    parser = argparse.ArgumentParser(
        description="Parse arguments for Upside-Down-RL-Transformer"
    )

    experiment_id = str(uuid.uuid4().hex)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["NeuralNet", "DecisionTransformer", "BERT_UDRL", "BERT_MLP"],
        default="NeuralNet",
    )
    parser.add_argument("--episodes", type=int, default=15)

    if training:
        parser.add_argument("--epochs", type=int, default=15)
        parser.add_argument("--hidden_size", type=int, default=256)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--learning_rate", type=float, default=3e-4)
        parser.add_argument("--patience", type=int, default=2)
    else:
        parser.add_argument("--d_r_array_length", type=int, default=36)

    args = parser.parse_args()

    set_seed(args.seed)

    if training:
        hyperparameters_dict = {
            "experiment_id": experiment_id,
            "model_type": args.model_type,
            "hidden_size": args.hidden_size,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "patience": args.patience,
            "episodes": args.episodes,
            "epochs": args.epochs
        }
    else:
        hyperparameters_dict = {
            "experiment_id": experiment_id,
            "model_type": args.model_type,
            "episodes": args.episodes,
            "d_r_array_length": args.d_r_array_length
        }

    return hyperparameters_dict
