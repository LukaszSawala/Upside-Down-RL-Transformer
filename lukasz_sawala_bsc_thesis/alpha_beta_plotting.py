import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import beta

def plot_beta_distribution(a=5, b=1, num_samples=10000, title="Beta Distribution", save_path="beta_distribution_plot.png"):
    """
    Plots the Beta distribution for given parameters a and b with a style similar to the `plot_average_rewards`.
    
    Args:
        a (float): Alpha parameter of the Beta distribution.
        b (float): Beta parameter of the Beta distribution.
        num_samples (int): Number of samples to plot.
        title (str): Title of the plot.
        save_path (str): Path to save the plot.
    """
    # Generate x values and Beta distribution y values
    x = np.linspace(0, 1, num_samples)
    y = beta.pdf(x, a, b)

    # Plotting setup
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    plt.figure(figsize=(12, 7))

    # Line plot for the Beta distribution
    sns.lineplot(x=x, y=y, linewidth=2.5, color="royalblue", marker="o", label=f"Beta({a}, {b})")

    # Fill the area under the curve
    plt.fill_between(x, y, color='royalblue', alpha=0.2)

    # Add a diagonal reference line
    plt.plot(x, x, linestyle="dotted", color="gray", label="y = x")

    # Labels and title
    plt.xlabel("Sampled Value (Normalized)", fontsize=12)
    plt.ylabel("Probability Density", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    sns.despine()

    # Save the plot
    plt.legend()
    plt.savefig(save_path)
    print(f"Beta distribution plot saved in {save_path}")

# Example usage
plot_beta_distribution(5, 1, title="Right-Skewed Beta Distribution", save_path="beta_right_skewed.png")
