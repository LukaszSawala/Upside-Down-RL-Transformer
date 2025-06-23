import torch
import itertools
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models import AntMazeBERTPretrainedMazeWrapper, AntMazeNNPretrainedMazeWrapper
from transfer_eval_main import (
    load_antmaze_nn_model_for_eval, load_antmaze_bertmlp_model_for_eval
)
from finetuningNN_maze import create_datasets
from grid_UDRLT_training_OPTIMIZED import set_seed, create_dataloaders

# ==== Paths ====
from dataset_generation import (
    INITIAL_ANTMAZE_BERT_PATH,
    NEW_BERT_MODEL_PATH,
    INITIAL_ANTMAZE_NN_PATH,
    NEW_NN_MODEL_PATH,
)
ROLLOUT_DATA_PATH = "antmaze_rollout_current_dataset.hdf5"

# ==== Configuration ====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE = 10
ACT_DIM = 8


def train_one_epoch(model, train_loader: DataLoader, optimizer: optim.Optimizer,
                    loss_fn: nn.Module, epoch_num: int, total_epochs: int) -> float:
    """
    Trains the model for one epoch.

    Args:
        model: The model to train.
        train_loader: The data loader for the training set.
        optimizer: The optimizer to use.
        loss_fn: The loss function to use.
        epoch_num: The current epoch number.
        total_epochs: The total number of epochs.

    Returns:
        The average training loss over the training set.
    """
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
    """
    Validates the model on the validation set.

    Args:
        model: The model to validate.
        val_loader: The data loader for the validation set.
        loss_fn: The loss function to use.
        epoch_num: The current epoch number.
        total_epochs: The total number of epochs.
        test_set: Whether to use the test set or the validation set.

    Returns:
        The average validation loss over the validation set.
    """
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
    """
    Trains the model for the specified number of epochs.

    Args:
        learning_rate: The learning rate to use.
        epochs: The number of epochs to train for.
        train_loader: The data loader for the training set.
        val_loader: The data loader for the validation set.
        model_to_use: The model to use.
        start_from_condition4: Whether to start from condition 4, used in the first iteration of the loop

    Returns:
        A dictionary containing the best model.
    """
    if model_to_use == "ANTMAZE_BERT_MLP":
        if start_from_condition4:
            model_components = load_antmaze_bertmlp_model_for_eval(INITIAL_ANTMAZE_BERT_PATH, DEVICE)
            model = AntMazeBERTPretrainedMazeWrapper(*model_components).to(DEVICE)
        else:
            model_components = load_antmaze_bertmlp_model_for_eval("", DEVICE, initialize_from_scratch=True)
            model = AntMazeBERTPretrainedMazeWrapper(*model_components).to(DEVICE)
            checkpoint = torch.load(NEW_BERT_MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(checkpoint["model"])
    elif model_to_use == "ANTMAZE_NN":
        if start_from_condition4:
            model = load_antmaze_nn_model_for_eval(INITIAL_ANTMAZE_NN_PATH, DEVICE)
            model = AntMazeNNPretrainedMazeWrapper(model).to(DEVICE)
        else:
            model = load_antmaze_nn_model_for_eval("", DEVICE,
                                                   initialize_from_scratch=True)
            model = AntMazeNNPretrainedMazeWrapper(model).to(DEVICE)
            checkpoint = torch.load(NEW_NN_MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(checkpoint["model"])

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
    train_ds, val_ds, test_ds = create_datasets(padding=False, data_path=ROLLOUT_DATA_PATH)  # do not pad with 0s

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
            save_path = NEW_BERT_MODEL_PATH if model_to_use == "ANTMAZE_BERT_MLP" else NEW_NN_MODEL_PATH
            torch.save(model_save_dict, save_path)
            print(f"Model for this configuration saved to {save_path}")

    print("\nGrid Search Complete.")
    return current_best_models["model"]


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    batch_sizes_param = [16]
    learning_rates_param = [5e-5]
    epochs_list_param = [100]
    start_from_condition4 = True  # Set to True at the beggining of the loop

    # Choose between "ANTMAZE_BERT_MLP" or "ANTMAZE_NN"
    model_to_use = "ANTMAZE_BERT_MLP"
    # model_to_use = "ANTMAZE_NN"

    _ = grid_search_experiment_from_rollout(batch_sizes_param=batch_sizes_param,
                                            learning_rates_param=learning_rates_param,
                                            epochs_list_param=epochs_list_param,
                                            model_to_use=model_to_use,
                                            start_from_condition4=start_from_condition4)
