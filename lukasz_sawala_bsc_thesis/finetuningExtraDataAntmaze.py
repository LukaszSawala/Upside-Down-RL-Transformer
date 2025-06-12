import gymnasium as gym
import torch
import numpy as np
import time
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import gymnasium_robotics
from scipy.stats import sem
import torch.nn as nn
from models import AntMazeBERTPretrainedMazeWrapper, AntMazeNNPretrainedMazeWrapper
from transfer_eval_main import (
    ANTMAZE_BERT_PATH, ANTMAZE_NN_PATH,
    load_antmaze_nn_model_for_eval, load_antmaze_bertmlp_model_for_eval
)
from finetuningNN_maze import create_datasets
from grid_UDRLT_training_OPTIMIZED import set_seed, create_dataloaders


# ==== Configuration ====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE = 10
ACT_DIM = 8


# ==== Paths ====
DATA_PATH = "../data/processed/antmaze_diverse_medium_concatenated_data.hdf5"
BEST_MODEL_PATH = "finetunedbroskiMERGED-512.pth"


def train_one_epoch(model, train_loader: DataLoader, optimizer: optim.Optimizer,
                    loss_fn: nn.Module, epoch_num: int, total_epochs: int) -> float:
    model.train()
    total_train_loss = 0.0

    print(f"Epoch {epoch_num}/{total_epochs} [Train]: Starting...")
    for (s, r, t, g, a) in train_loader:  # state, reward, time, goal, action
        s, r, t, g, a = s.to(DEVICE), r.to(DEVICE), t.to(DEVICE), g.to(DEVICE), a.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        action_tensor = model(s, r, t, g, DEVICE, use_goal=True)
        pred = action_tensor.squeeze(0).cpu().numpy()
        loss = loss_fn(pred, a)

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    return total_train_loss / len(train_loader)


def validate_one_epoch(model, val_loader: DataLoader, loss_fn: nn.Module,
                       epoch_num: int, total_epochs: int, test_set: bool = False) -> float:
    model.eval()
    total_val_loss = 0.0

    if test_set:
        print(f"Epoch {epoch_num}/{total_epochs} [Test ]: Starting...")
    else:
        print(f"Epoch {epoch_num}/{total_epochs} [Val  ]: Starting...")
    with torch.no_grad():
        for (s, r, t, g, a) in val_loader:  # state, reward, time, goal, action
            s, r, t, g, a = s.to(DEVICE), r.to(DEVICE), t.to(DEVICE), g.to(DEVICE), a.to(DEVICE)
            action_tensor = model(s, r, t, g, DEVICE, use_goal=True)
            pred = action_tensor.squeeze(0).cpu().numpy()
            loss = loss_fn(pred, a)
            total_val_loss += loss.item()
    return total_val_loss / len(val_loader)


def train_model(learning_rate: float, epochs: int, train_loader: DataLoader,
                val_loader: DataLoader, model_to_use) -> dict | None:
    if model_to_use == "ANTMAZE_BERT_MLP":
        model_components = load_antmaze_bertmlp_model_for_eval(ANTMAZE_BERT_PATH, DEVICE)
        model = AntMazeBERTPretrainedMazeWrapper(*model_components).to(DEVICE)
    elif model_to_use == "ANTMAZE_NN":
        model = load_antmaze_nn_model_for_eval(ANTMAZE_NN_PATH, DEVICE)
        model = AntMazeNNPretrainedMazeWrapper(model).to(DEVICE)

    optimizer = optim.Adam(list(model.parameters()), lr=learning_rate)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = PATIENCE
    current_best_models = None

    for epoch in range(epochs):
        avg_train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, epoch + 1, epochs
        )
        avg_val_loss = validate_one_epoch(
            model, val_loader, loss_fn, epoch + 1, epochs
        )

        print(
            f"Epoch {epoch + 1}/{epochs}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = PATIENCE
            current_best_models = {
                "model": model,
            }
            print(f"Best model found! Validation Loss: {best_val_loss:.4f}")
        else:
            patience_counter -= 1
            if patience_counter == 0:
                print("Early stopping.")
                break

    return current_best_models


def evaluate_model(
    model_to_evaluate: dict | None,
    test_loader: DataLoader,
) -> float:
    """
    Evaluates the model on the test set using provided state dictionaries.

    Args:
        model_state_dicts: Dictionary containing state_dicts of model components.
        test_loader: DataLoader for the test set.

    Returns:
        The average loss over the test set. Returns float('inf') if no model state is provided.
    """
    if model_to_evaluate is None:
        print("Error: No model state provided for evaluation.")
        return float('inf')

    model = model_to_evaluate["model"]
    model.to(DEVICE)
    loss_fn = nn.MSELoss()
    model.eval()

    print("Evaluation: Starting...")
    return validate_one_epoch(
        model, test_loader, loss_fn, 1, 1, test_set=True
    )
    

# ==== Grid Search ====
def grid_search_experiment() -> None:
    """
    Performs a grid search over specified hyperparameters.
    For each combination, it trains a model and saves the best version (based on its
    own validation loss during that run) to BEST_MODEL_PATH, overwriting previous saves.
    An evaluation on the test set is performed and printed for each model trained.
    """
    batch_sizes_param = [128]
    learning_rates_param = [1e-4]
    epochs_list_param = [100]

    # Choose between "ANTMAZE_BERT_MLP" or "ANTMAZE_NN"
    model_to_use = "ANTMAZE_BERT_MLP"
    #model_to_use = "ANTMAZE_NN"  

    param_grid = itertools.product(batch_sizes_param, learning_rates_param, epochs_list_param)

    train_ds, val_ds, test_ds = create_datasets()

    overall_best_test_loss = float("inf")
    overall_best_config_str = "None"

    for current_batch_size, current_lr, current_epochs in param_grid:
        current_config_str = (
            f"BATCH_SIZE={current_batch_size}, LEARNING_RATE={current_lr}, EPOCHS={current_epochs}"
        )
        print(f"\nRunning grid search with {current_config_str}")

        train_loader, val_loader, test_loader = create_dataloaders(
            train_ds, val_ds, test_ds, batch_size=current_batch_size
        )

        current_best_models = train_model(
            current_lr, current_epochs, train_loader, val_loader, model_to_use=model_to_use
        )

        if current_best_models:
            print(f"Training complete for {current_config_str}. Evaluating on test set...")
            current_test_loss = evaluate_model(current_best_models, test_loader)
            print(f"Test Loss for config ({current_config_str}): {current_test_loss:.4f}")

            if current_test_loss < overall_best_test_loss:
                overall_best_test_loss = current_test_loss
                overall_best_config_str = current_config_str
                model_save_dict = {
                    "model": current_best_models["model"].state_dict()
                }
                torch.save(model_save_dict, BEST_MODEL_PATH)
                print(f"Model for this configuration saved to {BEST_MODEL_PATH}")

        print("=" * 60)

    print("\nGrid Search Complete.")
    if overall_best_test_loss != float('inf'):
        print(f"The configuration that yielded the best reported test loss was: {overall_best_config_str}")
        print(f"Best reported Test Loss during search: {overall_best_test_loss:.4f}")
    print(f"The model saved at '{BEST_MODEL_PATH}' corresponds to the results of the *last* successfully trained hyperparameter set.")


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    print(f"Creating a model: {BEST_MODEL_PATH}")
    set_seed(42)
    grid_search_experiment()
