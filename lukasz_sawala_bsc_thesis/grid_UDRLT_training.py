import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import AutoModel, AutoConfig
from tqdm import tqdm
import h5py
from utils import set_seed


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE = 2
DATA_PATH = "../data/processed/concatenated_data.hdf5"
BEST_MODEL_PATH = "best_bert_udrl.pth"
STATE_DIM = 105
ACT_DIM = 8


def _load_data():
    with h5py.File(DATA_PATH, "r") as f:
        data = f["concatenated_data"]
        states = data["observations"][:]
        actions = data["actions"][:]
        rewards = data["rewards_to_go"][:].reshape(-1, 1)
        times = data["time_to_go"][:].reshape(-1, 1)
    return states, rewards, times, actions  # times referred to as "horizon" later on and in the paper to ensure consistency


def create_datasets():
    X_s, X_r, X_h, y = _load_data()
    X_s, X_r, X_h, y = map(torch.tensor, (X_s, X_r, X_h, y))
    dataset = TensorDataset(X_s.float(), X_r.float(), X_h.float(), y.float())
    lengths = [int(len(dataset) * 0.8), int(len(dataset) * 0.1)]
    lengths.append(len(dataset) - sum(lengths))
    train_ds, val_ds, test_ds = random_split(
        dataset, lengths, generator=torch.Generator().manual_seed(42)
    )
    return train_ds, val_ds, test_ds


def create_dataloaders(train_ds, val_ds, test_ds, batch_size=16):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(42))
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    return train_loader, val_loader, test_loader


def initiate_UDRLt_model():
    # Load untrained BERT-small
    config = AutoConfig.from_pretrained("prajjwal1/bert-small")
    config.vocab_size = 1  # dummy since we're using inputs_embeds
    config.max_position_embeddings = 3
    model_bert = AutoModel.from_config(config).to(DEVICE)

    # Create input projection layers and head
    d_r_encoder = nn.Linear(1, config.hidden_size).to(DEVICE)
    d_h_encoder = nn.Linear(1, config.hidden_size).to(DEVICE)
    state_encoder = nn.Linear(STATE_DIM, config.hidden_size).to(DEVICE)
    head = nn.Linear(config.hidden_size, ACT_DIM).to(DEVICE)
    return model_bert, d_r_encoder, d_h_encoder, state_encoder, head


def train(learning_rate, epochs, train_loader, val_loader):
    model_bert, d_r_encoder, d_h_encoder, state_encoder, head = initiate_UDRLt_model()
    optimizer = optim.Adam(
        list(model_bert.parameters())
        + list(d_r_encoder.parameters())
        + list(d_h_encoder.parameters())
        + list(state_encoder.parameters())
        + list(head.parameters()),
        lr=learning_rate,
    )
    loss_fn = nn.MSELoss()

    best_loss = float("inf")
    patience = PATIENCE

    for epoch in range(epochs):
        model_bert.train()
        total_train_loss = 0.0
        for s, r, h, a in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            s, r, h, a = s.to(DEVICE), r.to(DEVICE), h.to(DEVICE), a.to(DEVICE)
            optimizer.zero_grad()
            encoded_r = d_r_encoder(r).unsqueeze(1)
            encoded_h = d_h_encoder(h).unsqueeze(1)
            encoded_s = state_encoder(s).unsqueeze(1)
            sequence = torch.cat([encoded_r, encoded_h, encoded_s], dim=1)
            bert_out = model_bert(inputs_embeds=sequence).last_hidden_state
            pred = head(bert_out[:, -1])
            loss = loss_fn(pred, a)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model_bert.eval()
        encoded_s.eval()
        encoded_r.eval()
        encoded_h.eval()
        head.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for s, r, h, a in val_loader:
                s, r, h, a = s.to(DEVICE), r.to(DEVICE), h.to(DEVICE), a.to(DEVICE)
                encoded_r = d_r_encoder(r).unsqueeze(1)
                encoded_h = d_h_encoder(h).unsqueeze(1)
                encoded_s = state_encoder(s).unsqueeze(1)
                sequence = torch.cat([encoded_r, encoded_h, encoded_s], dim=1)
                bert_out = model_bert(inputs_embeds=sequence).last_hidden_state
                pred = head(bert_out[:, -1])
                loss = loss_fn(pred, a)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)

        print(
            f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience = PATIENCE
            current_best_model = {
                "bert": model_bert.state_dict(),
                "d_r": d_r_encoder.state_dict(),
                "d_h": d_h_encoder.state_dict(),
                "state": state_encoder.state_dict(),
                "head": head.state_dict(),
            }
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping.")
                break
    return current_best_model


def evaluate(model_state, test_loader):
    """Evaluates the UDRLt model on the test set."""
    model_bert, d_r_encoder, d_h_encoder, state_encoder, head = initiate_UDRLt_model()
    loss_fn = nn.MSELoss()

    # Load model weights
    model_bert.load_state_dict(model_state["bert"])
    d_r_encoder.load_state_dict(model_state["d_r"])
    d_h_encoder.load_state_dict(model_state["d_h"])
    state_encoder.load_state_dict(model_state["state"])
    head.load_state_dict(model_state["head"])

    # Set to evaluation mode
    model_bert.eval()
    d_r_encoder.eval()
    d_h_encoder.eval()
    state_encoder.eval()
    head.eval()

    total_loss = 0.0

    with torch.no_grad():
        for s, r, h, a in test_loader:
            s, r, h, a = s.to(DEVICE), r.to(DEVICE), h.to(DEVICE), a.to(DEVICE)

            encoded_r = d_r_encoder(r).unsqueeze(1)
            encoded_h = d_h_encoder(h).unsqueeze(1)
            encoded_s = state_encoder(s).unsqueeze(1)

            sequence = torch.cat([encoded_r, encoded_h, encoded_s], dim=1)
            bert_out = model_bert(inputs_embeds=sequence).last_hidden_state

            pred = head(bert_out[:, -1])
            loss = loss_fn(pred, a)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    return avg_loss


def grid_search():
    batch_sizes = [16, 8]
    learning_rates = [1e-4, 5e-5]
    epochs_list = [10, 20]
    param_grid = itertools.product(batch_sizes, learning_rates, epochs_list)

    best_test_loss = float("inf")
    best_config = None

    # Grid Search
    for BATCH_SIZE, LEARNING_RATE, EPOCHS in param_grid:
        print(
            f"Running grid search with BATCH_SIZE={BATCH_SIZE}, LEARNING_RATE={LEARNING_RATE}, EPOCHS={EPOCHS}"
        )
        train_ds, val_ds, test_ds = create_datasets()
        train_loader, val_loader, test_loader = create_dataloaders(
            train_ds, val_ds, test_ds, batch_size=BATCH_SIZE
        )
        model = train(LEARNING_RATE, EPOCHS, train_loader, val_loader)
        test_loss = evaluate(model, test_loader)

        if test_loss < best_test_loss:
            print("new best model found !!!!!!!!!!!!!!!!!!")
            best_test_loss = test_loss
            best_config = {
                "batch_size": BATCH_SIZE,
                "lr": LEARNING_RATE,
                "epochs": EPOCHS,
            }
            torch.save(model, BEST_MODEL_PATH)

        print(f"Test Loss for this config: {test_loss:.4f}")
        print("=" * 50)

    print(f"\nBest Config: {best_config}, Best Test Loss: {best_test_loss:.4f}")


if __name__ == "__main__":
    set_seed(42)
    grid_search()
