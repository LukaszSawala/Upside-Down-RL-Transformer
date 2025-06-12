import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
from transformers import DecisionTransformerModel, DecisionTransformerConfig

# --------- CONFIG ---------
MODEL_PATH = "../models/best_DT_grid_now_plt1.pth"
DATA_PATH = "../data/processed/episodic_data.hdf5"
MAX_LEN = 60
STATE_DIM = 105
ACT_DIM = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample_episode_window(file_path: str, max_len: int = 60) -> dict:
    data = h5py.File(file_path, 'r')['episodic_data']
    episodes = list(data.keys())
    idx = np.random.randint(0, len(episodes))
    episode = data[episodes[idx]]

    T = episode['observations'].shape[0]
    start = 0
    end = min(T, max_len)

    states = episode['observations'][start:end]
    actions = episode['actions'][start:end]
    rtg = episode['rewards_to_go'][start:end + 1] if end + 1 <= T else np.vstack([episode['rewards_to_go'][start:], np.zeros((1, 1))])
    ts = np.arange(start, end).reshape(-1, 1)

    pad_len = max_len - states.shape[0]
    if pad_len > 0:
        states = np.vstack([np.zeros((pad_len, states.shape[1])), states])
        actions = np.vstack([np.ones((pad_len, actions.shape[1])) * -10., actions])
        rtg = np.vstack([np.zeros((pad_len, 1)), rtg])
        ts = np.vstack([np.zeros((pad_len, 1)), ts])
    else:
        rtg = rtg[:max_len + 1]

    mask = (np.arange(max_len) >= pad_len).astype(np.float32)

    return {
        'states': torch.tensor(states, dtype=torch.float32).unsqueeze(0),
        'actions': torch.tensor(actions, dtype=torch.float32).unsqueeze(0),
        'rtgs': torch.tensor(rtg[:-1], dtype=torch.float32).unsqueeze(0),
        'timesteps': torch.tensor(ts.squeeze(), dtype=torch.long).unsqueeze(0),
        'attention_mask': torch.tensor(mask, dtype=torch.float32).unsqueeze(0),
    }


def make_token_labels(max_len: int) -> list:
    labels = []
    for t in range(max_len):
        labels.extend([f"s_{t}", f"a_{t}", f"r_{t+1}"])
    return labels[:3 * max_len]


def visualize_attention_importance_avg(model, num_batches: int = 100, top_k: int = 15):
    token_importances = None

    for _ in range(num_batches):
        batch = sample_episode_window(DATA_PATH, max_len=MAX_LEN)
        inputs = {k: v.to(DEVICE) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(
                states=inputs['states'],
                actions=inputs['actions'],
                rewards=None,
                returns_to_go=inputs['rtgs'],
                timesteps=inputs['timesteps'],
                attention_mask=inputs['attention_mask'],
                output_attentions=True,
            )

        attn = outputs.attentions[0]
        avg_attn = attn.mean(dim=1).squeeze(0).cpu().numpy()
        mean_attention = avg_attn.mean(axis=0)  # Attention received by each token

        if token_importances is None:
            token_importances = mean_attention
        else:
            token_importances += mean_attention

    token_importances /= num_batches
    token_labels = make_token_labels(MAX_LEN)

    # Get top-k tokens
    sorted_idx = np.argsort(token_importances)[::-1]
    top_indices = sorted_idx[:top_k]
    top_labels = [token_labels[i] for i in top_indices]
    top_scores = token_importances[top_indices]
    colors = ['red' if label.startswith('r_') else 'royalblue' for label in top_labels]

    plt.figure(figsize=(10, 5))
    plt.barh(top_labels[::-1], top_scores[::-1], color=colors[::-1])
    plt.xlabel("Avg Attention Received")
    plt.ylabel("Token")
    plt.title(f"Top-{top_k} Most Attended Tokens (Averaged over {num_batches} Batches)")
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("attention_importance_DT_top20.png")
    plt.show()


if __name__ == "__main__":
    print("using device ", DEVICE)
    config = DecisionTransformerConfig(
        state_dim=STATE_DIM,
        act_dim=ACT_DIM,
        max_length=MAX_LEN,
    )
    model = DecisionTransformerModel(config)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    visualize_attention_importance_avg(model, num_batches=10000, top_k=20)
