from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py


if __name__ == "__main__":
    # Load the dataset
    file_path = "../data/processed/concatenated_data.hdf5"
    with h5py.File(file_path, "r") as f:
        observations = np.array(f["concatenated_data/observations"])
        rewards_to_go = np.array(f["concatenated_data/rewards_to_go"])

    num_clusters = 100  # Adjust based on dataset size
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    state_clusters = kmeans.fit_predict(observations)

    # Aggregate rewards-to-go for each cluster
    clustered_dr = {i: [] for i in range(num_clusters)}
    for idx, cluster in enumerate(state_clusters):
        clustered_dr[cluster].append(rewards_to_go[idx])

    plt.figure(figsize=(12, 6))
    for cluster_id, dr_values in clustered_dr.items():
        sns.kdeplot(dr_values, label=f"Cluster {cluster_id}", fill=True)

    plt.xlabel("Rewards-to-Go (DR)", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.title("Distribution of Rewards-to-Go Across Similar State Clusters", fontsize=16, weight="bold")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.savefig("rewards_to_go_per_cluster.png")
