import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoConfig, AutoModel

from models import ScalarEncoder, ActionHead
from utils import set_seed


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "../data/processed/concatenated_data.hdf5"
MODEL_PATH = "bert_t_augm_enc_froz_action.pth"


def _load_data():
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


def load_test_loader(batch_size=16):
    X_s, X_r, X_h, y = _load_data()
    dataset = torch.utils.data.TensorDataset(X_s, X_r, X_h, y)
    lengths = [int(len(dataset) * 0.8), int(len(dataset) * 0.1)]
    lengths.append(len(dataset) - sum(lengths))
    _, _, test_ds = random_split(
        dataset, lengths, generator=torch.Generator().manual_seed(42)
    )
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False)


def load_bert_udrl_model(state_dim, act_dim, checkpoint_path):
    config = AutoConfig.from_pretrained("prajjwal1/bert-tiny")
    config.vocab_size = 1
    config.max_position_embeddings = 3
    config.output_attentions = True

    model_bert = AutoModel.from_config(config).to(DEVICE)
    d_r_encoder = ScalarEncoder(config.hidden_size).to(DEVICE)
    d_h_encoder = ScalarEncoder(config.hidden_size).to(DEVICE)
    state_encoder = nn.Linear(state_dim, config.hidden_size).to(DEVICE)
    head = ActionHead(config.hidden_size, act_dim).to(DEVICE)

    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model_bert.load_state_dict(ckpt["bert"])
    d_r_encoder.load_state_dict(ckpt["d_r"])
    d_h_encoder.load_state_dict(ckpt["d_h"])
    state_encoder.load_state_dict(ckpt["state"])
    head.load_state_dict(ckpt["head"])

    return model_bert.eval(), d_r_encoder.eval(), d_h_encoder.eval(), state_encoder.eval()


def visualize_attention(model_bert, d_r_encoder, d_h_encoder, state_encoder, test_loader):
    total_attention = torch.zeros(3, device=DEVICE)  # reward, horizon, state
    count = 0

    with torch.no_grad():
        for s, r, h, _ in test_loader:
            s, r, h = s.to(DEVICE), r.to(DEVICE), h.to(DEVICE)
            emb_r = d_r_encoder(r).unsqueeze(1)
            emb_h = d_h_encoder(h).unsqueeze(1)
            emb_s = state_encoder(s).unsqueeze(1)
            x = torch.cat([emb_r, emb_h, emb_s], dim=1)

            output = model_bert(inputs_embeds=x)

            #attn = torch.stack(output.attentions)  # (num_layers, batch, heads, tokens, tokens)
            #attn = attn.mean(dim=0).mean(dim=1)     # (batch, tokens, tokens) mean over layers & heads

            attn = output.attentions[0]        # attention from first layer only
            attn = attn.mean(dim=1)   

            # Sum attention *received* by each token
            attention_received = attn.mean(dim=1).sum(dim=0)  # (tokens,)
            total_attention += attention_received
            count += attn.shape[0]

    avg_attention = (total_attention / count).cpu().numpy()
    plt.figure(figsize=(6, 4))
    sns.barplot(x=["reward", "horizon", "state"], y=avg_attention, palette="magma")
    plt.ylabel("Average Attention Received")
    plt.title(f"Token-Level Attention (averaged over {count} samples)")
    plt.tight_layout()
    plt.savefig("attention_importance_udrl.png")
    plt.close()


if __name__ == "__main__":
    print("Using device:", DEVICE)
    set_seed(42)
    test_loader = load_test_loader()
    model_bert, d_r_encoder, d_h_encoder, state_encoder = load_bert_udrl_model(
        state_dim=105, act_dim=8, checkpoint_path=MODEL_PATH
    )
    visualize_attention(model_bert, d_r_encoder, d_h_encoder, state_encoder, test_loader)
