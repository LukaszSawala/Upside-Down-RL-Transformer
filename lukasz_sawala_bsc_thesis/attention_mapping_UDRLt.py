import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import AutoConfig, AutoModel
from utils import set_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "../data/processed/concatenated_data.hdf5"
MODEL_PATH = "../models/bert_s_grid.pth"


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
        horizons = data["time_to_go"][:].reshape(-1, 1)
    return (
        torch.tensor(states).float(),
        torch.tensor(rewards).float(),
        torch.tensor(horizons).float(),
        torch.tensor(actions).float(),
    )


def load_test_loader(batch_size: int = 16) -> DataLoader:
    """
    Creates the data loader for the test set, following the
    exact same logic as done in the training loop to ensure reproducibility.
    """
    X_s, X_r, X_h, y = _load_data()
    dataset = torch.utils.data.TensorDataset(X_s, X_r, X_h, y)
    lengths = [int(len(dataset) * 0.8), int(len(dataset) * 0.1)]
    lengths.append(len(dataset) - sum(lengths))
    _, _, test_ds = random_split(
        dataset, lengths, generator=torch.Generator().manual_seed(42)
    )
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False)


def load_model() -> tuple:
    """
    Loads the BERT model for the attention mapping task.

    Returns:
        tuple: A tuple containing:
            - model_bert (AutoModel): The BERT model.
            - hidden_size (int): The hidden size of the BERT model.
    """
    config = AutoConfig.from_pretrained("prajjwal1/bert-small")
    config.vocab_size = 1
    config.max_position_embeddings = 3
    config.output_attentions = True  # crucial for this task
    model_bert = AutoModel.from_config(config).to(DEVICE)
    return model_bert, config.hidden_size


def visualize_attention(
    model_bert: AutoModel,
    d_r_encoder: nn.Linear,
    d_h_encoder: nn.Linear,
    state_encoder: nn.Linear,
    test_loader: torch.utils.data.DataLoader,
) -> None:
    """
    Computes and plots the average attention received by each input token
    (reward, horizon, state) across all layers, heads, and batches.

    Saves a bar plot with one score per variable.
    """
    model_bert.eval()
    d_r_encoder.eval()
    d_h_encoder.eval()
    state_encoder.eval()

    total_attention = torch.zeros(3, device=DEVICE)  # for reward, horizon, state
    count = 0

    with torch.no_grad():
        for i, (s, r, h, _) in enumerate(test_loader):
            s, r, h = s.to(DEVICE), r.to(DEVICE), h.to(DEVICE)
            emb_r = d_r_encoder(r).unsqueeze(1)
            emb_h = d_h_encoder(h).unsqueeze(1)
            emb_s = state_encoder(s).unsqueeze(1)
            x = torch.cat([emb_r, emb_h, emb_s], dim=1)

            output = model_bert(inputs_embeds=x)
            attn = torch.stack(output.attentions)
            attn = attn.mean(dim=0).mean(dim=1)

            # For each sample in batch, accumulate attention RECEIVED by each token
            attention_received = attn.mean(dim=1).sum(dim=0)
            total_attention += attention_received
            count += attn.shape[0]

    avg_attention = (total_attention / count).detach().cpu().numpy()
    tokens = ["reward", "horizon", "state"]
    plt.figure(figsize=(6, 4))
    sns.barplot(x=tokens, y=avg_attention, palette="viridis")
    plt.ylabel("Average Attention Received")
    plt.title(f"Token-Level Attention (averaged over {count} samples)")
    plt.tight_layout()
    plt.savefig("attention_importance_udrl.png")
    plt.close()


if __name__ == "__main__":
    print("using device ", DEVICE)
    set_seed(42)
    test_loader = load_test_loader()
    model_bert, hidden_size = load_model()
    d_r_encoder = torch.nn.Linear(1, hidden_size).to(DEVICE)
    d_h_encoder = torch.nn.Linear(1, hidden_size).to(DEVICE)
    state_encoder = torch.nn.Linear(105, hidden_size).to(DEVICE)

    model_state = torch.load(MODEL_PATH)
    model_bert.load_state_dict(model_state["bert"])
    d_r_encoder.load_state_dict(model_state["d_r"])
    d_h_encoder.load_state_dict(model_state["d_t"])
    state_encoder.load_state_dict(model_state["state"])

    visualize_attention(
        model_bert, d_r_encoder, d_h_encoder, state_encoder, test_loader
    )
