import matplotlib.pyplot as plt
import numpy as np


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
        plt.savefig("layerwise_evolution.pdf", format="pdf", dpi=300)

    plt.show()
