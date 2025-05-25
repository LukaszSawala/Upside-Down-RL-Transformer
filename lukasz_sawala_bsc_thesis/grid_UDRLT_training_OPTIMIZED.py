#==============================================================================
# This file contains the training code for the Upside-Down-RL-Transformer model.
# It is quite complex due to optimization methods applied to speed up the training
# process. For a better reference on the training algorithm, see
# grid_UDRLT_MLP_training.py, following a similar logic with some modifications.
#==============================================================================

import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import AutoModel, AutoConfig
from tqdm import tqdm
import h5py
from utils import set_seed
from models import ActionHead 


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE = 2
DATA_PATH = "../data/processed/concatenated_data.hdf5"
BEST_MODEL_PATH = "frozen_actionhead_best_bert_udrl.pth" 
STATE_DIM = 105
ACT_DIM = 8

"""
Test Loss for config (BATCH_SIZE=8, LEARNING_RATE=1e-05, EPOCHS=15): 0.0167 (no action head, small)
"""

def _load_data() -> tuple:
    """
    This function reads the concatenated dataset from the specified HDF5 file 
    and extracts the observations, actions, rewards, and time-to-go values. 
    The rewards and time-to-go values are reshaped to be 2D arrays. 
    All data is converted to PyTorch tensors with float dtype.

    Returns:
        tuple: A tuple containing:
            - states (torch.Tensor): Observations of shape (N, STATE_DIM).
            - rewards (torch.Tensor): Rewards-to-go of shape (N, 1).
            - horizons (torch.Tensor): Time-to-go of shape (N, 1).
            - actions (torch.Tensor): Actions of shape (N, ACT_DIM).
    """
    with h5py.File(DATA_PATH, "r") as f:
        data = f["concatenated_data"]
        states = data["observations"][:]
        actions = data["actions"][:]
        rewards = data["rewards_to_go"][:].reshape(-1, 1)
        times = data["time_to_go"][:].reshape(-1, 1)
    return states, rewards, times, actions


def create_datasets() -> tuple:
    """
    Creates train, validation, and test datasets from the concatenated dataset.
    Data is read from the HDF5 file, converted to PyTorch tensors, and split into 80% train, 10% validation, and 10% test sets.
    Returns:
        tuple: A tuple containing:
            - train_ds (TensorDataset): The training dataset.
            - val_ds (TensorDataset): The validation dataset.
            - test_ds (TensorDataset): The test dataset.
    """
    X_s_np, X_r_np, X_h_np, y_np = _load_data()
    X_s = torch.tensor(X_s_np, dtype=torch.float32)
    X_r = torch.tensor(X_r_np, dtype=torch.float32)
    X_h = torch.tensor(X_h_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)
    
    dataset = TensorDataset(X_s, X_r, X_h, y)
    
    lengths = [int(len(dataset) * 0.8), int(len(dataset) * 0.1)]
    lengths.append(len(dataset) - sum(lengths))
    train_ds, val_ds, test_ds = random_split(
        dataset, lengths, generator=torch.Generator().manual_seed(42)
    )
    return train_ds, val_ds, test_ds


def create_dataloaders(train_ds: TensorDataset, val_ds: TensorDataset,
                       test_ds: TensorDataset, batch_size: int = 16,
                       num_workers: int = 0) -> tuple:
    """
    Create DataLoader objects for the train, validation, and test datasets.
    Args:
        train_ds (Dataset): The training dataset.
        val_ds (Dataset): The validation dataset.
        test_ds (Dataset): The test dataset.
        batch_size (int, optional): Number of samples per batch. Defaults to 16.
        num_workers (int, optional): How many subprocesses to use for data loading.
                                     0 means that the data will be loaded in the main process.
                                     Defaults to 0.

    Returns:
        tuple: A tuple containing:
            - train_loader (DataLoader): DataLoader for the training dataset.
            - val_loader (DataLoader): DataLoader for the validation dataset.
            - test_loader (DataLoader): DataLoader for the test dataset.
    """
    pin_memory = True if DEVICE.type == 'cuda' else False

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, generator=torch.Generator().manual_seed(42),
                              num_workers=num_workers, pin_memory=pin_memory
                              )
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader


def initiate_UDRLt_model() -> tuple:
    """
    Initializes the Upside-Down Reinforcement Learning Transformer (UDRLt) model components.

    This function configures and initializes a BERT model and several linear layers
    for encoding the reward, horizon, and state, as well as a linear head for action prediction.

    Returns:
        tuple: A tuple containing:
            - model_bert (AutoModel): The initialized BERT model.
            - d_r_encoder (nn.Linear): Linear layer for encoding reward.
            - d_h_encoder (nn.Linear): Linear layer for encoding horizon.
            - state_encoder (nn.Linear): Linear layer for encoding state.
            - head (nn.Linear): Linear layer for predicting actions.
    """
    config = AutoConfig.from_pretrained("prajjwal1/bert-small")
    config.vocab_size = 1  # dummy since we're using inputs_embeds
    config.max_position_embeddings = 3 # R, H, S
    model_bert = AutoModel.from_config(config).to(DEVICE)

    d_r_encoder = nn.Linear(1, config.hidden_size).to(DEVICE)
    for param in d_r_encoder.parameters():
        param.requires_grad = False

    d_h_encoder = nn.Linear(1, config.hidden_size).to(DEVICE)
    for param in d_h_encoder.parameters():
        param.requires_grad = False

    state_encoder = nn.Linear(STATE_DIM, config.hidden_size).to(DEVICE)
    for param in state_encoder.parameters():
        param.requires_grad = False
    #head = nn.Linear(config.hidden_size, ACT_DIM).to(DEVICE)
    head = ActionHead(config.hidden_size, ACT_DIM).to(DEVICE)

    return model_bert, d_r_encoder, d_h_encoder, state_encoder, head


def train(learning_rate: float, epochs: int,
          train_loader: DataLoader, val_loader: DataLoader) -> dict:
    """
    Trains the Upside-Down Reinforcement Learning Transformer (UDRLt) model on the
    given data, using the specified learning rate and number of epochs.
    Uses Automatic Mixed Precision (AMP) if CUDA is available.

    Returns:
        dict: A dictionary containing the state_dicts of the best model components.
    """
    model_bert, d_r_encoder, d_h_encoder, state_encoder, head = initiate_UDRLt_model()

    optimizer = optim.Adam(
        list(model_bert.parameters())
        + list(head.parameters()),
        lr=learning_rate,
    )
    loss_fn = nn.MSELoss()

    # --- Automatic Mixed Precision (AMP) Setup ---
    scaler = None
    if DEVICE.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
    # --- End of AMP Setup ---

    best_loss = float("inf")
    patience_counter = PATIENCE
    current_best_model_state = None

    for epoch in range(epochs):
        # Training
        model_bert.train()
        head.train()
        total_train_loss = 0.0
        
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")
        for s, r, h, a in train_loop:
            s = s.to(DEVICE, non_blocking=True if train_loader.pin_memory else False)
            r = r.to(DEVICE, non_blocking=True if train_loader.pin_memory else False)
            h = h.to(DEVICE, non_blocking=True if train_loader.pin_memory else False)
            a = a.to(DEVICE, non_blocking=True if train_loader.pin_memory else False)

            optimizer.zero_grad(set_to_none=True)

            # --- AMP: Forward pass ---
            if scaler: # If using CUDA and AMP
                with torch.cuda.amp.autocast():
                    encoded_r = d_r_encoder(r).unsqueeze(1)
                    encoded_h = d_h_encoder(h).unsqueeze(1)
                    encoded_s = state_encoder(s).unsqueeze(1)
                    sequence = torch.cat([encoded_r, encoded_h, encoded_s], dim=1)
                    bert_out = model_bert(inputs_embeds=sequence).last_hidden_state
                    pred = head(bert_out[:, -1]) # Use last token's output for prediction
                    loss = loss_fn(pred, a)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else: # If not using AMP (e.g., on CPU)
                encoded_r = d_r_encoder(r).unsqueeze(1)
                encoded_h = d_h_encoder(h).unsqueeze(1)
                encoded_s = state_encoder(s).unsqueeze(1)
                sequence = torch.cat([encoded_r, encoded_h, encoded_s], dim=1)
                bert_out = model_bert(inputs_embeds=sequence).last_hidden_state
                pred = head(bert_out[:, -1])
                loss = loss_fn(pred, a)
                loss.backward()
                optimizer.step()
            # --- End of AMP specific code ---
            
            total_train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model_bert.eval()
        head.eval()
        total_val_loss = 0.0
        
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val  ]")
        with torch.no_grad():
            for s, r, h, a in val_loop:
                s = s.to(DEVICE, non_blocking=True if val_loader.pin_memory else False)
                r = r.to(DEVICE, non_blocking=True if val_loader.pin_memory else False)
                h = h.to(DEVICE, non_blocking=True if val_loader.pin_memory else False)
                a = a.to(DEVICE, non_blocking=True if val_loader.pin_memory else False)

                # AMP for inference (though benefits are mainly in memory and speed if ops are fp16-friendly)
                if DEVICE.type == 'cuda': # autocast can be used in eval too
                     with torch.cuda.amp.autocast():
                        encoded_r = d_r_encoder(r).unsqueeze(1)
                        encoded_h = d_h_encoder(h).unsqueeze(1)
                        encoded_s = state_encoder(s).unsqueeze(1)
                        sequence = torch.cat([encoded_r, encoded_h, encoded_s], dim=1)
                        bert_out = model_bert(inputs_embeds=sequence).last_hidden_state
                        pred = head(bert_out[:, -1])
                        loss = loss_fn(pred, a)
                else:
                    encoded_r = d_r_encoder(r).unsqueeze(1)
                    encoded_h = d_h_encoder(h).unsqueeze(1)
                    encoded_s = state_encoder(s).unsqueeze(1)
                    sequence = torch.cat([encoded_r, encoded_h, encoded_s], dim=1)
                    bert_out = model_bert(inputs_embeds=sequence).last_hidden_state
                    pred = head(bert_out[:, -1])
                    loss = loss_fn(pred, a)
                
                total_val_loss += loss.item()
                val_loop.set_postfix(loss=loss.item())
        avg_val_loss = total_val_loss / len(val_loader)

        print(
            f"Epoch {epoch + 1}/{epochs}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = PATIENCE
            current_best_model_state = {
                "bert": model_bert.state_dict(),
                "d_r": d_r_encoder.state_dict(),
                "d_h": d_h_encoder.state_dict(),
                "state": state_encoder.state_dict(),
                "head": head.state_dict(),
            }
        else:
            patience_counter -= 1
            if patience_counter == 0:
                print("Early stopping.")
                break
    
    if current_best_model_state is None and epochs > 0: # Handle case where no improvement was ever made but epochs ran
        print("Warning: No improvement in validation loss. Saving the last model state.")
        current_best_model_state = {
            "bert": model_bert.state_dict(),
            "d_r": d_r_encoder.state_dict(),
            "d_h": d_h_encoder.state_dict(),
            "state": state_encoder.state_dict(),
            "head": head.state_dict(),
        }
    elif epochs == 0: # Should not happen with typical setup but good to cover
        print("Warning: Zero epochs requested.")
        return None


    return current_best_model_state


def evaluate(model_state: dict, test_loader: torch.utils.data.DataLoader) -> float:
    """
    Evaluates the UDRLt model on the test set.
    Uses Automatic Mixed Precision (AMP) for inference if CUDA is available.
    Returns:
        float: The average loss over the test set.
    """
    if model_state is None:
        print("Error: No model state provided for evaluation.")
        return float('inf')

    model_bert, d_r_encoder, d_h_encoder, state_encoder, head = initiate_UDRLt_model()
    loss_fn = nn.MSELoss()

    model_bert.load_state_dict(model_state["bert"])
    d_r_encoder.load_state_dict(model_state["d_r"])
    d_h_encoder.load_state_dict(model_state["d_h"])
    state_encoder.load_state_dict(model_state["state"])
    head.load_state_dict(model_state["head"])

    model_bert.eval()
    d_r_encoder.eval()
    d_h_encoder.eval()
    state_encoder.eval()
    head.eval()

    total_loss = 0.0
    
    test_loop = tqdm(test_loader, desc="Evaluating")
    with torch.no_grad():
        for s, r, h, a in test_loop:
            s = s.to(DEVICE, non_blocking=True if test_loader.pin_memory else False)
            r = r.to(DEVICE, non_blocking=True if test_loader.pin_memory else False)
            h = h.to(DEVICE, non_blocking=True if test_loader.pin_memory else False)
            a = a.to(DEVICE, non_blocking=True if test_loader.pin_memory else False)

            if DEVICE.type == 'cuda':
                 with torch.cuda.amp.autocast():
                    encoded_r = d_r_encoder(r).unsqueeze(1)
                    encoded_h = d_h_encoder(h).unsqueeze(1)
                    encoded_s = state_encoder(s).unsqueeze(1)
                    sequence = torch.cat([encoded_r, encoded_h, encoded_s], dim=1)
                    bert_out = model_bert(inputs_embeds=sequence).last_hidden_state
                    pred = head(bert_out[:, -1])
                    loss = loss_fn(pred, a)
            else:
                encoded_r = d_r_encoder(r).unsqueeze(1)
                encoded_h = d_h_encoder(h).unsqueeze(1)
                encoded_s = state_encoder(s).unsqueeze(1)
                sequence = torch.cat([encoded_r, encoded_h, encoded_s], dim=1)
                bert_out = model_bert(inputs_embeds=sequence).last_hidden_state
                pred = head(bert_out[:, -1])
                loss = loss_fn(pred, a)
            
            total_loss += loss.item()
            test_loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(test_loader)
    return avg_loss


def grid_search() -> None:
    """
    Performs a grid search over the given hyperparameters and saves the best model found.
    Saves the best model found to BEST_MODEL_PATH and prints out the best config and best test loss.
    """
    batch_sizes = [8] # Original: [16, 8]
    learning_rates = [5e-5] # Original: [1e-4, 5e-5]
    epochs_list = [15] # Original: [10, 20]
    
    # If your CPU usage is low during training, increase num_workers.
    # If it's maxed out and causing issues, decrease it.
    num_data_workers = 8 if DEVICE.type == 'cuda' else 0
    param_grid = itertools.product(batch_sizes, learning_rates, epochs_list)

    best_test_loss = float("inf")
    best_config = None
    train_ds, val_ds, test_ds = create_datasets()

    for BATCH_SIZE, LEARNING_RATE, EPOCHS in param_grid:
        current_config_str = f"BATCH_SIZE={BATCH_SIZE}, LEARNING_RATE={LEARNING_RATE}, EPOCHS={EPOCHS}"
        print(f"\nRunning grid search with {current_config_str}")
        
        train_loader, val_loader, test_loader = create_dataloaders(
            train_ds, val_ds, test_ds, batch_size=BATCH_SIZE, num_workers=num_data_workers
        )
        
        model_state = train(LEARNING_RATE, EPOCHS, train_loader, val_loader)
        
        if model_state is None:
            print(f"Training failed or was skipped for config: {current_config_str}. Skipping evaluation.")
            continue

        test_loss = evaluate(model_state, test_loader)

        print(f"Test Loss for config ({current_config_str}): {test_loss:.4f}")

        if test_loss < best_test_loss:
            print(f"New best model found! Test Loss: {test_loss:.4f} (Improved from {best_test_loss:.4f})")
            best_test_loss = test_loss
            best_config = {
                "batch_size": BATCH_SIZE,
                "lr": LEARNING_RATE,
                "epochs": EPOCHS,
            }
            torch.save(model_state, BEST_MODEL_PATH)
            print(f"Best model saved to {BEST_MODEL_PATH}")

        print("=" * 60)

    if best_config:
        print(f"\nGrid Search Complete.\nBest Config: {best_config}, Best Test Loss: {best_test_loss:.4f}")


if __name__ == "__main__":
    print("Using device:", DEVICE)
    set_seed(42)
    grid_search()
