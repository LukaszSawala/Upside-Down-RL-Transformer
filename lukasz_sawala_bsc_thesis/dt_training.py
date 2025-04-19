import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from transformers import DecisionTransformerModel, DecisionTransformerConfig

# --- CONFIGURATION ---
STATE_DIM = 105       # antv5 observation dim
ACT_DIM = 8           # antv5 action dim
MAX_LENGTH = 20       # DT context window
BATCH_SIZE = 16
LR = 1e-4
TOTAL_STEPS = 1000
GRAD_CLIP = 0.25
DATA_PATH = "../data/processed/episodic_data.hdf5"
DT_MODEL_PATH = "../models/best_DT.pth"
TRAIN_EPISODES = 900

# --- DATASET CLASS ---
class EpisodicHDF5Dataset(Dataset):
    def __init__(self, file_path, max_len=MAX_LENGTH, train=True, train_episodes=TRAIN_EPISODES):
        self.data = h5py.File(file_path, 'r')['episodic_data']
        all_episodes = list(self.data.keys())
        self.episodes = all_episodes[:train_episodes] if train else all_episodes[train_episodes:]
        self.max_len = max_len

    def __len__(self):
        return len(self.episodes)

    def sample_window(self):
        idx = np.random.randint(0, len(self.episodes))
        episode = self.data[self.episodes[idx]]

        T = episode['observations'].shape[0]
        if T >= self.max_len:
            start = np.random.randint(0, T - self.max_len + 1)
            end = start + self.max_len
        else:
            start = 0
            end = T

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
            'rtgs': torch.tensor(rtg[:-1], dtype=torch.float32),  # Use rtg[t-1]
            'timesteps': torch.tensor(ts.squeeze(), dtype=torch.long),
            'action_target': torch.tensor(actions, dtype=torch.float32),
            'attention_mask': torch.tensor(mask, dtype=torch.float32),
        }

    def __getitem__(self, idx):
        return self.sample_window()

# --- MODEL SETUP ---
config = DecisionTransformerConfig(
    state_dim=STATE_DIM,
    act_dim=ACT_DIM,
    max_length=MAX_LENGTH,
)
model = DecisionTransformerModel(config)
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

# --- TRAINING LOOP ---
def train(model, train_dataloader):
    model.train()
    for step in range(TOTAL_STEPS):
        total_loss = 0
        for batch in train_dataloader:
            states = batch['states']
            actions = batch['actions']
            rtgs = batch['rtgs']
            timesteps = batch['timesteps']
            attention_mask = batch['attention_mask']
            targets = batch['action_target']

            outputs = model(
                states=states,
                actions=actions,
                rewards=None,
                returns_to_go=rtgs,
                timesteps=timesteps,
                attention_mask=attention_mask,
            )

            action_preds = outputs['action_preds'] # The model outputs action, state and reward preds 
            mask = attention_mask.unsqueeze(-1).expand_as(action_preds) > 0

            loss = loss_fn(action_preds[mask], targets[mask])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            total_loss += loss.item()
        if step + 1 % 100 == 0:
            print(f"Timestep {step+1}/{TOTAL_STEPS}, Loss: {total_loss / len(train_dataloader):.4f}")

# --- DATA LOADING & TRAINING ---
train_dataset = EpisodicHDF5Dataset(DATA_PATH, train=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = EpisodicHDF5Dataset(DATA_PATH, train=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

train(model, train_loader)