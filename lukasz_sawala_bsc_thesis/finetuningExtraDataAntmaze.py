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
import torch.nn as nn
from models import AntMazeBERTPretrainedMazeWrapper, AntMazeNNPretrainedMazeWrapper
from transfer_eval_main import (
    load_antmaze_nn_model_for_eval, load_antmaze_bertmlp_model_for_eval
)
from finetuningNN_maze import create_datasets
from grid_UDRLT_training_OPTIMIZED import set_seed, create_dataloaders


# ==== Configuration ====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE = 10
ACT_DIM = 8


# ==== Paths ====
ROLLOUT_DATA_PATH = "antmaze_rollout_current_dataset.hdf5"
from dataset_generation import (
    INITIAL_ANTMAZE_BERT_PATH, NEW_MODEL_PATH,
    INITIAL_ANTMAZE_NN_PATH
)

def train_one_epoch(model, train_loader: DataLoader, optimizer: optim.Optimizer,
                    loss_fn: nn.Module, epoch_num: int, total_epochs: int) -> float:
    model.train()
    total_train_loss = 0.0

    print(f"Epoch {epoch_num}/{total_epochs} [Train]: Starting...")
    for (s, r, t, g, a) in train_loader:  # state, reward, time, goal, action
        s, r, t, g, a = s.to(DEVICE), r.to(DEVICE), t.to(DEVICE), g.to(DEVICE), a.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        pred = model(s, r, t, g, DEVICE, use_goal=True)
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
            pred = model(s, r, t, g, DEVICE, use_goal=True)
            loss = loss_fn(pred, a)
            total_val_loss += loss.item()
    return total_val_loss / len(val_loader)


def train_model(learning_rate: float, epochs: int, train_loader: DataLoader,
                val_loader: DataLoader, model_to_use: str, start_from_condition4: bool) -> dict | None:
    if model_to_use == "ANTMAZE_BERT_MLP":
        if start_from_condition4:
            model_components = load_antmaze_bertmlp_model_for_eval(INITIAL_ANTMAZE_BERT_PATH, DEVICE)
        else:
            model_components = load_antmaze_bertmlp_model_for_eval(NEW_MODEL_PATH, DEVICE)
        model = AntMazeBERTPretrainedMazeWrapper(*model_components).to(DEVICE)
    elif model_to_use == "ANTMAZE_NN":
        if start_from_condition4:
            model = load_antmaze_nn_model_for_eval(INITIAL_ANTMAZE_NN_PATH, DEVICE)
        else:
            model = load_antmaze_nn_model_for_eval(NEW_MODEL_PATH, DEVICE)
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
    

# ==== Grid Search ====
def grid_search_experiment_from_rollout(batch_sizes_param: list, learning_rates_param: list,
                           epochs_list_param: list, model_to_use: str,
                           start_from_condition4: bool) -> AntMazeBERTPretrainedMazeWrapper:
    """
    Performs a grid search over specified hyperparameters.
    For each combination, it trains a model and saves the best version (based on its
    own validation loss during that run) to BEST_MODEL_PATH, overwriting previous saves.
    An evaluation on the test set is performed and printed for each model trained.
    Returns the best model from the last iteration of the grid search.
    """
    set_seed(42)
    param_grid = itertools.product(batch_sizes_param, learning_rates_param, epochs_list_param)
    train_ds, val_ds, test_ds = create_datasets(padding=False, data_path=ROLLOUT_DATA_PATH) # do not pad with 0s

    for current_batch_size, current_lr, current_epochs in param_grid:
        current_config_str = (
            f"BATCH_SIZE={current_batch_size}, LEARNING_RATE={current_lr}, EPOCHS={current_epochs}"
        )
        print(f"\nRunning grid search with {current_config_str}")

        train_loader, val_loader, _ = create_dataloaders(
            train_ds, val_ds, test_ds, batch_size=current_batch_size
        )

        current_best_models = train_model(
            current_lr, current_epochs, train_loader, val_loader, model_to_use=model_to_use, 
            start_from_condition4=start_from_condition4
        )

        if current_best_models:
            model_save_dict = {
                "model": current_best_models["model"].state_dict()
            }
            torch.save(model_save_dict, NEW_MODEL_PATH)
            print(f"Model for this configuration saved to {NEW_MODEL_PATH}")

    print("\nGrid Search Complete.")
    return current_best_models["model"]


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    print(f"Creating a model: {NEW_MODEL_PATH}")
    batch_sizes_param = [16]
    learning_rates_param = [5e-5]
    epochs_list_param = [100]
    start_from_condition4 = True  # Set to True at the beggining of the loop

    # Choose between "ANTMAZE_BERT_MLP" or "ANTMAZE_NN"
    model_to_use = "ANTMAZE_BERT_MLP"
    #model_to_use = "ANTMAZE_NN"  

    _ = grid_search_experiment_from_rollout(batch_sizes_param=batch_sizes_param,
                           learning_rates_param=learning_rates_param,
                           epochs_list_param=epochs_list_param,
                           model_to_use=model_to_use,
                           start_from_condition4=start_from_condition4)
