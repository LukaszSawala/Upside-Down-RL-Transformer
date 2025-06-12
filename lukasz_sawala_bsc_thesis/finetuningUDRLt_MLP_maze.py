import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoModel
from model_evaluation import (
    load_bert_mlp_model_for_eval, BERT_MLP_MODEL_PATH
)
from grid_UDRLT_training_OPTIMIZED import set_seed, create_dataloaders
from models import BertAntMazeActionHead
from finetuningNN_maze import create_datasets

# ==== Configuration ====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE = 10
ACT_DIM = 8
set_seed(42)


# ==== Paths ====
DATA_PATH = "../data/processed/antmaze_diverse_medium_concatenated_data.hdf5"
BEST_MODEL_PATH = "finetunedbroski-512.pth"

# ======================================= FILE EXPLANATION ======================================

# This script is designed to fine-tune a pre-trained Ant UDRLt-MLP model using an AntMaze dataset from the 
# Farama Foundation (medium-diverse) (condition 2 of the research).

# ================================================================================================

def train_one_epoch(model_bert: AutoModel, state_encoder: nn.Linear, mlp_head,
                    final_actionhead: BertAntMazeActionHead, train_loader: DataLoader, optimizer: optim.Optimizer,
                    loss_fn: nn.Module, epoch_num: int, total_epochs: int) -> float:
    """
    Trains the model for one epoch.
    Args:
        model_bert: The BERT model.
        state_encoder: The linear layer to project state to BERT input size.
        mlp_head: The MLP head to predict the base output.
        final_actionhead: The final action head to predict the action.
        train_loader: The data loader for the training set.
        optimizer: The optimizer to use.
        loss_fn: The loss function to use.
        epoch_num: The current epoch number.
        total_epochs: The total number of epochs.

    Returns:
        The average training loss over the training set.
    """
    model_bert.train()
    state_encoder.train()
    mlp_head.train()
    final_actionhead.train()
    total_train_loss = 0.0

    print(f"Epoch {epoch_num}/{total_epochs} [Train]: Starting...")
    for (s, r, t, g, a) in train_loader:  # state, reward, time, goal, action
        s, r, t, g, a = s.to(DEVICE), r.to(DEVICE), t.to(DEVICE), g.to(DEVICE), a.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        # old model
        s_proj = state_encoder(s).unsqueeze(1)
        bert_out = model_bert(inputs_embeds=s_proj).last_hidden_state[:, 0]
        input_to_mlp = torch.cat([bert_out, r, t], dim=1)
        base_output = mlp_head(input_to_mlp)
        # new action head
        final_input = torch.cat([base_output, g], dim=1)  # add the goal information
        pred = final_actionhead(final_input)
        loss = loss_fn(pred, a)

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    return total_train_loss / len(train_loader)


def validate_one_epoch(model_bert: AutoModel, state_encoder: nn.Linear, mlp_head,
                       final_actionhead: BertAntMazeActionHead, val_loader: DataLoader, loss_fn: nn.Module,
                       epoch_num: int, total_epochs: int) -> float:
    """
    Validates the model for one epoch.

    Args:
        model_bert: The BERT model.
        state_encoder: The linear layer to project state to BERT input size.
        mlp_head: The MLP head to predict the base output.
        final_actionhead: The final action head to predict the action.
        val_loader: The data loader for the validation set.
        loss_fn: The loss function to use.
        epoch_num: The current epoch number.
        total_epochs: The total number of epochs.

    Returns:
        The average validation loss over the validation set.
    """
    model_bert.eval()
    state_encoder.eval()
    mlp_head.eval()
    final_actionhead.eval()
    total_val_loss = 0.0

    print(f"Epoch {epoch_num}/{total_epochs} [Val  ]: Starting...")
    with torch.no_grad():
        for (s, r, t, g, a) in val_loader:  # state, reward, time, goal, action
            s, r, t, g, a = s.to(DEVICE), r.to(DEVICE), t.to(DEVICE), g.to(DEVICE), a.to(DEVICE)
            # old model
            s_proj = state_encoder(s).unsqueeze(1)
            bert_out = model_bert(inputs_embeds=s_proj).last_hidden_state[:, 0]
            input_to_mlp = torch.cat([bert_out, r, t], dim=1)
            base_output = mlp_head(input_to_mlp)
            # new action head
            final_input = torch.cat([base_output, g], dim=1)  # add the goal information
            pred = final_actionhead(final_input)
            loss = loss_fn(pred, a)

            total_val_loss += loss.item()
    return total_val_loss / len(val_loader)


def train_model(learning_rate: float, epochs: int, train_loader: DataLoader, val_loader: DataLoader) -> dict | None:
    """
    Trains the model using the specified hyperparameters.
    Implements early stopping based on validation loss.

    Args:
        learning_rate: The learning rate for the optimizer.
        epochs: The number of epochs to train for.
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.

    Returns:
        A dictionary containing the state_dicts of the best model components,
        or None if training did not produce a best model (e.g., 0 epochs).
    """
    model_bert, state_encoder, mlp_head = load_bert_mlp_model_for_eval(BERT_MLP_MODEL_PATH, DEVICE, freeze=True)
    action_head = BertAntMazeActionHead(hidden_size=512, act_dim=ACT_DIM, num_layers=12).to(DEVICE)

    optimizer = optim.Adam(list(action_head.parameters()), lr=learning_rate)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = PATIENCE
    current_best_models = None

    for epoch in range(epochs):
        avg_train_loss = train_one_epoch(
            model_bert, state_encoder, mlp_head, action_head, train_loader, optimizer, loss_fn, epoch + 1, epochs
        )
        avg_val_loss = validate_one_epoch(
            model_bert, state_encoder, mlp_head, action_head, val_loader, loss_fn, epoch + 1, epochs
        )

        print(
            f"Epoch {epoch + 1}/{epochs}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = PATIENCE
            current_best_models = {
                "bert": model_bert,
                "state": state_encoder,
                "mlp": mlp_head,
                "action_head": action_head,
            }
            print(f"Best model found! Validation Loss: {best_val_loss:.4f}")
        else:
            patience_counter -= 1
            if patience_counter == 0:
                print("Early stopping.")
                break

    return current_best_models


def evaluate_model(
    models_to_evaluate: dict | None,
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
    if models_to_evaluate is None:
        print("Error: No model state provided for evaluation.")
        return float('inf')

    model_bert, state_encoder, mlp_head, action_head = models_to_evaluate["bert"], models_to_evaluate["state"], models_to_evaluate["mlp"], models_to_evaluate["action_head"]
    model_bert.to(DEVICE)
    state_encoder.to(DEVICE)
    mlp_head.to(DEVICE)
    action_head.to(DEVICE)

    loss_fn = nn.MSELoss()

    model_bert.eval()
    state_encoder.eval()
    mlp_head.eval()
    action_head.eval()

    total_test_loss = 0.0

    print("Evaluation: Starting...")
    with torch.no_grad():
        for (s, r, t, g, a) in test_loader:  # state, reward, time, goal, action
            s, r, t, g, a = s.to(DEVICE), r.to(DEVICE), t.to(DEVICE), g.to(DEVICE), a.to(DEVICE)
            # old model
            s_proj = state_encoder(s).unsqueeze(1)
            bert_out = model_bert(inputs_embeds=s_proj).last_hidden_state[:, 0]
            input_to_mlp = torch.cat([bert_out, r, t], dim=1)
            base_output = mlp_head(input_to_mlp)
            # new action head
            final_input = torch.cat([base_output, g], dim=1)  # add the goal information
            pred = action_head(final_input)
            loss = loss_fn(pred, a)

            total_test_loss += loss.item()
    avg_test_loss = total_test_loss / len(test_loader)
    return avg_test_loss


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
            current_lr, current_epochs, train_loader, val_loader
        )

        if current_best_models:
            print(f"Training complete for {current_config_str}. Evaluating on test set...")
            current_test_loss = evaluate_model(current_best_models, test_loader)
            print(f"Test Loss for config ({current_config_str}): {current_test_loss:.4f}")

            if current_test_loss < overall_best_test_loss:
                overall_best_test_loss = current_test_loss
                overall_best_config_str = current_config_str
                models_save_dict = {
                    "bert": current_best_models["bert"].state_dict(),
                    "state": current_best_models["state"].state_dict(),
                    "mlp": current_best_models["mlp"].state_dict(),
                    "action_head": current_best_models["action_head"].state_dict(),
                }
                torch.save(models_save_dict, BEST_MODEL_PATH)
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
