import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import AutoModel, AutoConfig
from tqdm import tqdm
import h5py
import numpy as np
from models import NeuralNet
from grid_UDRLT_training_OPTIMIZED import create_datasets, create_dataloaders

# ==== Configuration ====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE = 10
DATA_PATH = "../data/processed/concatenated_data.hdf5"
BEST_MODEL_PATH = "new-architecture-berttiny-batch32.pth" # From original script
STATE_DIM = 105 # Derived from original state_encoder input
ACT_DIM = 8


def set_seed(seed_value: int) -> None:
    """Sets the seed for reproducibility."""
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def initiate_model_components() -> tuple[AutoModel, nn.Linear, NeuralNet]:
    """
    Initializes the model components based on the original script's architecture.
    This includes a BERT model for state embedding, a state encoder, and an MLP.

    Returns:
        tuple: model_bert, state_encoder, mlp_head.
    """
    config = AutoConfig.from_pretrained("prajjwal1/bert-tiny")
    config.vocab_size = 1
    config.max_position_embeddings = 1
    model_bert = AutoModel.from_config(config).to(DEVICE)

    state_encoder = nn.Linear(STATE_DIM, config.hidden_size).to(DEVICE)

    final_input_size = config.hidden_size + 2
    mlp_head = HugeNeuralNet(input_size=final_input_size, hidden_size=256, output_size=ACT_DIM).to(DEVICE)

    return model_bert, state_encoder, mlp_head


# ==== Training and Evaluation ====
def train_one_epoch(model_bert: AutoModel, state_encoder: nn.Linear, mlp_head: NeuralNet,
    train_loader: DataLoader, optimizer: optim.Optimizer, loss_fn: nn.Module,
    epoch_num: int, total_epochs: int) -> float:
    """Trains the model for one epoch."""
    model_bert.train()
    state_encoder.train()
    mlp_head.train()
    total_train_loss = 0.0

    
    print(f"Epoch {epoch_num}/{total_epochs} [Train]: Starting...")
    for (s, r, t, a) in train_loader:
        s, r, t, a = s.to(DEVICE), r.to(DEVICE), t.to(DEVICE), a.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)

        s_proj = state_encoder(s).unsqueeze(1)  # [B, 1, hidden]
        bert_out = model_bert(inputs_embeds=s_proj).last_hidden_state[:, 0] # [B, hidden]
        input_to_mlp = torch.cat([bert_out, r, t], dim=1) # [B, hidden+2]
        pred = mlp_head(input_to_mlp)
        loss = loss_fn(pred, a)

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    return total_train_loss / len(train_loader)


def validate_one_epoch(model_bert: AutoModel, state_encoder: nn.Linear, mlp_head: NeuralNet,
    val_loader: DataLoader, loss_fn: nn.Module, epoch_num: int, total_epochs: int) -> float:
    """Validates the model for one epoch."""
    model_bert.eval()
    state_encoder.eval()
    mlp_head.eval()
    total_val_loss = 0.0

    print(f"Epoch {epoch_num}/{total_epochs} [Val  ]: Starting...")
    with torch.no_grad():
        for (s, r, t, a) in val_loader:
            s, r, t, a = s.to(DEVICE), r.to(DEVICE), t.to(DEVICE), a.to(DEVICE)
            s_proj = state_encoder(s).unsqueeze(1)
            bert_out = model_bert(inputs_embeds=s_proj).last_hidden_state[:, 0]
            input_to_mlp = torch.cat([bert_out, r, t], dim=1)
            pred = mlp_head(input_to_mlp)
            loss = loss_fn(pred, a)

            total_val_loss += loss.item()
    return total_val_loss / len(val_loader)


def train_model(learning_rate: float, epochs: int, train_loader: DataLoader,val_loader: DataLoader) -> dict | None:
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
    model_bert, state_encoder, mlp_head = initiate_model_components()

    optimizer = optim.Adam(
        list(model_bert.parameters())
        + list(state_encoder.parameters())
        + list(mlp_head.parameters()),
        lr=learning_rate,
    )
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = PATIENCE
    current_best_model_state_dicts = None

    for epoch in range(epochs):
        avg_train_loss = train_one_epoch(
            model_bert, state_encoder, mlp_head, train_loader, optimizer, loss_fn, epoch + 1, epochs
        )
        avg_val_loss = validate_one_epoch(
             model_bert, state_encoder, mlp_head, val_loader, loss_fn, epoch + 1, epochs
        )

        print(
            f"Epoch {epoch + 1}/{epochs}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = PATIENCE
            current_best_model_state_dicts = {
                "bert": model_bert.state_dict(),
                "state_encoder": state_encoder.state_dict(),
                "mlp_head": mlp_head.state_dict(),
            }
            print(f"Best model found! Validation Loss: {best_val_loss:.4f}")
        else:
            patience_counter -= 1
            if patience_counter == 0:
                print("Early stopping.")
                break

    return current_best_model_state_dicts


def evaluate_model(
    model_state_dicts: dict | None,
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
    if model_state_dicts is None:
        print("Error: No model state provided for evaluation.")
        return float('inf')

    model_bert, state_encoder, mlp_head = initiate_model_components()
    loss_fn = nn.MSELoss()

  
    model_bert.load_state_dict(model_state_dicts["bert"])
    state_encoder.load_state_dict(model_state_dicts["state_encoder"])
    mlp_head.load_state_dict(model_state_dicts["mlp_head"])

    model_bert.eval()
    state_encoder.eval()
    mlp_head.eval()

    total_test_loss = 0.0
    test_loop = tqdm(test_loader, desc="Evaluating")
    with torch.no_grad():
        for s, r, t, a in test_loop:
            s = s.to(DEVICE, non_blocking=True if test_loader.pin_memory else False)
            r = r.to(DEVICE, non_blocking=True if test_loader.pin_memory else False)
            t = t.to(DEVICE, non_blocking=True if test_loader.pin_memory else False)
            a = a.to(DEVICE, non_blocking=True if test_loader.pin_memory else False)

            s_proj = state_encoder(s).unsqueeze(1)
            bert_out = model_bert(inputs_embeds=s_proj).last_hidden_state[:, 0]
            input_to_mlp = torch.cat([bert_out, r, t], dim=1)
            pred = mlp_head(input_to_mlp)
            loss = loss_fn(pred, a)

            total_test_loss += loss.item()
            test_loop.set_postfix(loss=loss.item())

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
    batch_sizes_param = [16]
    learning_rates_param = [5e-5]
    epochs_list_param = [50]
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

        best_model_state_dicts_for_run = train_model(
            current_lr, current_epochs, train_loader, val_loader
        )

        if best_model_state_dicts_for_run:
            print(f"Training complete for {current_config_str}. Evaluating on test set...")
            current_test_loss = evaluate_model(best_model_state_dicts_for_run, test_loader)
            print(f"Test Loss for config ({current_config_str}): {current_test_loss:.4f}")

            torch.save(best_model_state_dicts_for_run, BEST_MODEL_PATH)
            print(f"Model for this configuration saved to {BEST_MODEL_PATH}")

            if current_test_loss < overall_best_test_loss:
                overall_best_test_loss = current_test_loss
                overall_best_config_str = current_config_str
        else:
            print(f"Training failed or was skipped for config: {current_config_str}. Skipping evaluation and save.")
        
        print("=" * 60)

    print("\nGrid Search Complete.")
    if overall_best_test_loss != float('inf'):
        print(f"The configuration that yielded the best reported test loss was: {overall_best_config_str}")
        print(f"Best reported Test Loss during search: {overall_best_test_loss:.4f}")
    else:
        print("No configurations were successfully trained and evaluated.")
    print(f"The model saved at '{BEST_MODEL_PATH}' corresponds to the results of the *last* successfully trained hyperparameter set.")


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    set_seed(42)
    grid_search_experiment()