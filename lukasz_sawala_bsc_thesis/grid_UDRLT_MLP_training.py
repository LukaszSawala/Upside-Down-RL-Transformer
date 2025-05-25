import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import AutoModel, AutoConfig
from tqdm import tqdm
import numpy as np
import h5py
from models import NeuralNet

# ==== Configuration ====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE = 3
BATCH_SIZE = 16
ACT_DIM = 8


# ==== Paths ====
# DATA_PATH = "extremely_augmented_data.hdf5"
DATA_PATH = "../data/processed/concatenated_data.hdf5"
BEST_MODEL_PATH = "new-architecture-berttiny-batch32.pth"


# ==== Data Loading ====
def load_data(path=DATA_PATH):
    with h5py.File(path, "r") as f:
        data = f["concatenated_data"]
        states = data["observations"][:]
        actions = data["actions"][:]
        rewards = data["rewards_to_go"][:].reshape(-1, 1)
        times = data["time_to_go"][:].reshape(-1, 1)
    return states, rewards, times, actions


def train_scalar_concat(BATCH_SIZE, LEARNING_RATE, EPOCHS, train_loader, val_loader):
    best_model = None
    # Load BERT config and model (for state only)
    config = AutoConfig.from_pretrained("prajjwal1/bert-tiny")
    config.vocab_size = 1
    config.max_position_embeddings = 1  # only state goes in
    model_bert = AutoModel.from_config(config).to(DEVICE)

    # State encoder to match BERT hidden size
    state_encoder = nn.Linear(105, config.hidden_size).to(DEVICE)

    # Final NN head that takes [bert_out (state), d_r, d_h] -> action
    final_input_size = config.hidden_size + 2
    mlp = NeuralNet(input_size=final_input_size, hidden_size=256, output_size=8).to(
        DEVICE
    )

    optimizer = optim.Adam(
        list(model_bert.parameters())
        + list(state_encoder.parameters())
        + list(mlp.parameters()),
        lr=LEARNING_RATE,
    )
    loss_fn = nn.MSELoss()

    best_loss = float("inf")
    patience = PATIENCE

    for epoch in range(EPOCHS):
        model_bert.train()
        state_encoder.train()
        mlp.train()
        total_train_loss = 0.0

        for s, r, t, a in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}"
        ):  # state, return-to-go, time-to-go (horizon), action
            s, r, t, a = s.to(DEVICE), r.to(DEVICE), t.to(DEVICE), a.to(DEVICE)
            optimizer.zero_grad()

            # Encode state with state encoder then BERT
            s_proj = state_encoder(s).unsqueeze(1)  # [B, 1, hidden]
            bert_out = model_bert(inputs_embeds=s_proj).last_hidden_state[
                :, 0
            ]  # [B, hidden]

            # Concatenate BERT state output with raw scalars
            input_to_mlp = torch.cat([bert_out, r, t], dim=1)  # [B, hidden+2]

            pred = mlp(input_to_mlp)
            loss = loss_fn(pred, a)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model_bert.eval()
        state_encoder.eval()
        mlp.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for s, r, t, a in val_loader:
                s, r, t, a = s.to(DEVICE), r.to(DEVICE), t.to(DEVICE), a.to(DEVICE)
                s_proj = state_encoder(s).unsqueeze(1)
                bert_out = model_bert(inputs_embeds=s_proj).last_hidden_state[:, 0]
                input_to_mlp = torch.cat([bert_out, r, t], dim=1)
                pred = mlp(input_to_mlp)
                loss = loss_fn(pred, a)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)

        print(
            f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience = PATIENCE
            best_model = {
                "bert": model_bert,
                "state": state_encoder,
                "mlp": mlp,
            }
            print("best model found !!!!!!!!!!!!!!!!!!")
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping.")
                break
    return best_model


def grid_search(train_loader, val_loader, test_loader):
    # Define hyperparameters grid
    batch_sizes = [16]  # dont change this lol doenst work
    learning_rates = [5e-5]
    epochs_list = [30]

    # Create combinations of all hyperparameters
    param_grid = itertools.product(batch_sizes, learning_rates, epochs_list)

    # Grid Search
    for BATCH_SIZE, LEARNING_RATE, EPOCHS in param_grid:
        print(
            f"Running grid search with BATCH_SIZE={BATCH_SIZE}, LEARNING_RATE={LEARNING_RATE}, EPOCHS={EPOCHS}"
        )
        best_model = train_scalar_concat(BATCH_SIZE, LEARNING_RATE, EPOCHS, train_loader, val_loader)
        # for now, don't evaluate on the test set but save the model
        torch.save(
            {
                "bert": best_model["bert"].state_dict(),
                "state": best_model["state"].state_dict(),
                "mlp": best_model["mlp"].state_dict(),
            },
            BEST_MODEL_PATH,
        )

if __name__ == "__main__":
    X_s, X_r, X_t, y = load_data()
    X_s, X_r, X_t, y = map(torch.tensor, (X_s, X_r, X_t, y))
    dataset = TensorDataset(X_s.float(), X_r.float(), X_t.float(), y.float())
    lengths = [int(len(dataset) * 0.8), int(len(dataset) * 0.1)]
    lengths.append(len(dataset) - sum(lengths))
    train_ds, val_ds, test_ds = random_split(
        dataset, lengths, generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    grid_search(train_loader, val_loader, test_loader)
