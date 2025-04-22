import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import h5py
import numpy as np
from transformers import DecisionTransformerModel, DecisionTransformerConfig
from itertools import product
from typing import Dict
from zeus.monitor import ZeusMonitor


# Best Config: {'batch_size': 16, 'lr': 0.0001, 'max_length': 30}, Best Test Loss: 0.0651


# --- CONFIGURATION ---
STATE_DIM = 105       # antv5 observation dim
ACT_DIM = 8           # antv5 action dim
TOTAL_EPOCHS = 20
GRAD_CLIP = 0.25
DATA_PATH = "../data/processed/episodic_data.hdf5"
DT_MODEL_PATH = "../models/best_DT_grid1.pth"

# --- DATASET CLASS ---
class EpisodicHDF5Dataset(Dataset):
    """
    Dataset class for loading episodic data in a form of a 
    context window from an HDF5 file.
    """
    def __init__(self, file_path: str, max_len: int = 20) -> None:        
        self.data = h5py.File(file_path, 'r')['episodic_data']
        self.episodes = list(self.data.keys())
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.episodes)

    def sample_window(self) -> Dict[str, torch.Tensor]: 
        """
        Sample a random context window of max_len from an episode 
        in the dataset and pad it to max_len if necessary.
        Note that this is a simplified version of the original sampling method
        taken from the Decision Transformer paper cited in the paper.
        Returns:
            - states: torch.tensor of shape (max_len, STATE_DIM)
            - actions: torch.tensor of shape (max_len, ACT_DIM)
            - rtgs: torch.tensor of shape (max_len, 1)
            - timesteps: torch.tensor of shape (max_len, 1)
            - action_target: torch.tensor of shape (max_len, ACT_DIM)
            - attention_mask: torch.tensor of shape (max_len,) indicating which timesteps are padded
        """
        idx = np.random.randint(0, len(self.episodes))
        episode = self.data[self.episodes[idx]]

        T = episode['observations'].shape[0]
        deviation_prob = 0.85  # Probability of getting a small end value
        if T >= self.max_len:
            start = np.random.randint(0, T - self.max_len + 1)
            end = start + self.max_len
            if np.random.rand() < deviation_prob:
                # Sample a smaller `end`, randomly between start + 1 and start + max_len 
                end = start + np.random.randint(1, int(self.max_len) + 1)
            else:
                # Otherwise, sample a normal end point close to start + max_len
                end = start + self.max_len
        else:
            start = 0
            end = T
        end = min(end, T)

        # Efficient slicing
        states = episode['observations'][start:end]
        actions = episode['actions'][start:end]
        rtg = episode['rewards_to_go'][start:end+1] if end + 1 <= T else np.vstack([episode['rewards_to_go'][start:], np.zeros((1, 1))])
        ts = np.arange(start, end).reshape(-1, 1)

        # Padding
        pad_len = self.max_len - states.shape[0]
        if pad_len > 0:
            states = np.vstack([np.zeros((pad_len, states.shape[1])), states])
            actions = np.vstack([np.ones((pad_len, actions.shape[1])) * -10., actions])
            rtg = np.vstack([np.zeros((pad_len, 1)), rtg])
            ts = np.vstack([np.zeros((pad_len, 1)), ts])
        else:
            rtg = rtg[:self.max_len + 1]  # +1 for rtg[t+1]

        mask = (np.arange(self.max_len) >= pad_len).astype(np.float32)

        return {
            'states': torch.tensor(states, dtype=torch.float32),
            'actions': torch.tensor(actions, dtype=torch.float32), 
            'rtgs': torch.tensor(rtg[:-1], dtype=torch.float32), 
            'timesteps': torch.tensor(ts.squeeze(), dtype=torch.long),
            'action_target': torch.tensor(actions, dtype=torch.float32),
            'attention_mask': torch.tensor(mask, dtype=torch.float32),
        }

    def __getitem__(self, idx):
        return self.sample_window()


def train(model, train_dataloader, val_dataloader, device, patience=2, lr=1e-4) -> tuple[Dict, float]:
    """
    Train the model on the given train dataloader and validate on the given val dataloader.
    
    Args:
    - model: the model to train
    - train_dataloader: the dataloader for the training data
    - val_dataloader: the dataloader for the validation data
    - device: the device to train on
    - patience: the patience for early stopping
    - lr: the learning rate
    
    Returns:
    - best_model_dict: the best model state dict
    - best_val_loss: the best validation loss
    """
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    best_val_loss = float('inf')
    best_model_dict = None

    for epoch in range(TOTAL_EPOCHS):
        # TRAINING ============================================
        total_train_loss = 0
        model.train()
        for batch in train_dataloader:
            states = batch['states'].to(device)
            actions = batch['actions'].to(device)
            rtgs = batch['rtgs'].to(device)
            timesteps = batch['timesteps'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['action_target'].to(device)

            outputs = model(
                states=states,
                actions=actions,
                rewards=None,
                returns_to_go=rtgs,
                timesteps=timesteps,
                attention_mask=attention_mask,
            )

            action_preds = outputs['action_preds']
            mask = attention_mask.unsqueeze(-1).expand_as(action_preds) > 0

            loss = loss_fn(action_preds[mask], targets[mask])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            total_train_loss += loss.item()

        # VALIDATION ===========================================
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                states = batch['states'].to(device)
                actions = batch['actions'].to(device)
                rtgs = batch['rtgs'].to(device)
                timesteps = batch['timesteps'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                targets = batch['action_target'].to(device)

                outputs = model(
                    states=states,
                    actions=actions,
                    rewards=None,
                    returns_to_go=rtgs,
                    timesteps=timesteps,
                    attention_mask=attention_mask,
                )

                action_preds = outputs['action_preds']
                mask = attention_mask.unsqueeze(-1).expand_as(action_preds) > 0
                val_loss = loss_fn(action_preds[mask], targets[mask])
                total_val_loss += val_loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"Epoch {epoch+1}/{TOTAL_EPOCHS}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience = 2
            best_model_dict = model.state_dict()
        else:
            patience -= 1
            if patience == 0:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Save the best model
    print(f"Best Val Loss for this config: {best_val_loss:.4f}")
    return best_model_dict, best_val_loss


def evaluate(model, test_loader, device) -> float:
    """
    Evaluate a model on a given test dataloader.

    Args:
        model: the model to evaluate
        test_loader: the test dataloader
        device: the device to run the evaluation on

    Returns:
        the average loss on the test set
    """
    model.eval()
    loss_fn = nn.MSELoss()
    total_loss = 0

    with torch.no_grad():
        for batch in test_loader:
            states = batch['states'].to(device)
            actions = batch['actions'].to(device)
            rtgs = batch['rtgs'].to(device)
            timesteps = batch['timesteps'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['action_target'].to(device)

            outputs = model(
                states=states,
                actions=actions,
                rewards=None,
                returns_to_go=rtgs,
                timesteps=timesteps,
                attention_mask=attention_mask,
            )

            action_preds = outputs['action_preds']
            mask = attention_mask.unsqueeze(-1).expand_as(action_preds) > 0
            loss = loss_fn(action_preds[mask], targets[mask])
            total_loss += loss.item()
    return total_loss / len(test_loader)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(42)
    print("using device:", device)
    search_space = {
        "batch_size": [8, 16],
        "lr": [5e-4, 1e-4],  # learning rate of the optimizer
        "max_length": [30, 40], # size of the context window for the DT
    }

    best_config = None
    best_test_loss = float('inf')

    #monitor = ZeusMonitor(device)
    #monitor.begin_window("grid-search-dt")

    # grid search definition
    for batch_size, lr, max_length in product(*search_space.values()):
        print(f"\nTesting config: batch_size={batch_size}, lr={lr}, max_length={max_length}")

        dataset = EpisodicHDF5Dataset(DATA_PATH, max_len=max_length)

        test_size = int(0.1 * len(dataset))
        val_size = int(0.1 * len(dataset))
        train_size = len(dataset) - test_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        config = DecisionTransformerConfig(
            state_dim=STATE_DIM,
            act_dim=ACT_DIM,
            max_length=max_length,  # size of the context window
        )
        model = DecisionTransformerModel(config).to(device)

        best_model_dict, val_loss = train(model, train_loader, val_loader, device, patience=3, lr=lr)
        model.load_state_dict(best_model_dict)
        test_loss = evaluate(model, test_loader, device)

        print(f"Test Loss for this config: {test_loss:.4f}")

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_config = {
                "batch_size": batch_size,
                "lr": lr,
                "max_length": max_length
            }
            torch.save(best_model_dict, DT_MODEL_PATH)

    #mes = monitor.end_window("grid-search")
    #print(f"Training grid search took {mes.time} s and consumed {mes.total_energy} J.")
    print(f"\nBest Config: {best_config}, Best Test Loss: {best_test_loss:.4f}")
