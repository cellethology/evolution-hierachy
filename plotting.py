import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap

def plot_layer_evolution(layer_stats, layers_to_plot=None, figsize=(8, 6), save=False):
    """
    Plot evolution of statistics across generations for multiple layers.

    Parameters
    ----------
    layer_stats : dict
        Dictionary containing 'mean', 'q1s', 'q3s' arrays of shape (n_layers, n_generations)
    layers_to_plot : list, optional
        List of layer indices to plot. If None, plots all layers
    figsize : tuple, optional
        Figure size
    """
    n_layers, n_generations = layer_stats["mean"].shape
    if layers_to_plot is None:
        layers_to_plot = range(n_layers)

    plt.figure(figsize=figsize)

    # Use different colors for different layers
    colors = plt.cm.viridis(np.linspace(0, 1, len(layers_to_plot)))
    # make the last color "tab:orange" in terms of four numbers
    colors[-1] = (1.0, 0.4980392156862745, 0.0, 1.0)

    for layer_idx, color in zip(layers_to_plot, colors):
        # Plot mean line
        plt.plot(
            range(n_generations),
            layer_stats["mean"][layer_idx],
            color=color,
            label=f"Layer {layer_idx}",
        )

        # Plot shaded area between +/- 1 standard deviations
        plt.fill_between(
            range(n_generations),
            layer_stats["mean"][layer_idx] - 1 * layer_stats["stdev"][layer_idx],
            layer_stats["mean"][layer_idx] + 1 * layer_stats["stdev"][layer_idx],
            color=color,
            alpha=0.2,
        )

    # Remove borders
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    # Remove grid
    plt.grid(False)

    # Increase font sizes
    plt.xlabel("Generation", fontsize=18)
    plt.ylabel("Optimality", fontsize=18)
    plt.tick_params(axis="both", labelsize=16)

    # Remove legend border and adjust position
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False, fontsize=16)
    plt.tight_layout()

    if save:
        plt.savefig("output/layerwise_evolution.pdf", format="pdf", dpi=300)

    plt.show()

def plot_stacked_ancestry_grid(
    ancestry_matrices,
    n_cols=3,
    figsize=(15, 8),
    titles=None,
    colormap="tab10",
    top_n_colored=10, 
    save=False):
    """
    ancestry_matrices: list of ancestry matrices (each of shape n_ancestors x n_generations)
    n_cols: number of columns in the grid
    figsize: overall figure size
    titles: optional list of titles for each subplot
    """
    n_plots = len(ancestry_matrices)
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for i, ancestry_matrix in enumerate(ancestry_matrices):
        ax = axes[i]

        n_ancestors, n_generations = ancestry_matrix.shape
        generations = range(n_generations)

        # Sort by final contribution for consistent layering
        total_contribution = ancestry_matrix.sum(axis=1)
        sorted_indices = np.argsort(-total_contribution)
        sorted_ancestry = ancestry_matrix[sorted_indices]

        # Create a colormap with unique colors for top N contributors
        cmap = get_cmap(colormap)
        colors = [None] * n_ancestors
        for j in range(n_ancestors):
            if j < top_n_colored:
                colors[j] = cmap(j / top_n_colored)  # assign distinct color
            else:
                colors[j] = (0.7, 0.7, 0.7, 0.3)  # gray-ish, semi-transparent

        ax.stackplot(generations, sorted_ancestry, colors=colors, alpha=0.95)
        ax.set_title(titles[i] if titles and i < len(titles) else f"Run {i+1}")
        ax.set_xlabel("generation")
        ax.set_ylabel("proportion")

        ax.grid(True, linestyle="--", alpha=0.4)

    # Turn off any unused subplots
    for j in range(n_plots, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Ancestry composition over time (across runs)", fontsize=22)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save:
        plt.savefig("output/ancestry_over_generation.jpeg", format="jpeg", dpi=300)
    plt.show()
