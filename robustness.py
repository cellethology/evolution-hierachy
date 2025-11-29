import matplotlib.pyplot as plt
import numpy as np
from layered_system import LayeredSystem, uniform_sphere_gaussian
from tqdm import tqdm


def measure_directional_robustness(
    max_depth,
    n_models=30,
    dim=20,
    n_samples=10000,
    perturbation_size=0.5,
    use_sigmoid=False,
    rank_fraction=1,
    seed=42,
):
    """
    Measure the directional robustness of a sequence of matrix operations.

    Parameters
    ----------
    max_depth : int
        Maximum depth of the random matrix product.
    n_models : int
        Number of random models to average over.
    dim : int
        Dimensionality of the input space.
    n_samples : int
        Number of samples to average over.
    perturbation_size : float
        Size of the perturbation to apply to the input samples.
    use_sigmoid : bool
        Whether to apply sigmoid activation after each matrix multiplication.
    rank_fraction : float
        Fraction of full rank for the matrices.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    medians, q1s, q3s : arrays
        Statistics of directional robustness for each depth.
    """
    np.random.seed(seed)

    # Generate fixed test vectors and perturbations once
    X = uniform_sphere_gaussian(n_samples, dim)
    noise = np.random.randn(n_samples, dim) * perturbation_size
    X_perturbed = X + noise

    # Store results for each depth across all models
    all_results = np.zeros((n_models, max_depth, n_samples))

    for model in tqdm(range(n_models)):
        # Create new system for each model
        system = LayeredSystem(
            dim=dim,
            max_depth=max_depth,
            rank_fraction=rank_fraction,
            use_sigmoid=use_sigmoid,
        )

        # Get layer outputs for both original and perturbed inputs
        outputs = system._forward_pass(X)
        outputs_perturbed = system._forward_pass(X_perturbed)

        # Compute cosine similarities at each depth
        for d in range(max_depth):
            Y = outputs[d]
            Y_perturbed = outputs_perturbed[d]

            norms = np.linalg.norm(Y, axis=1) * np.linalg.norm(Y_perturbed, axis=1)
            cosine_sims = 1 - np.abs(np.sum(Y * Y_perturbed, axis=1) / norms)
            all_results[model, d, :] = cosine_sims

    # Compute statistics across all models and samples
    medians = np.median(all_results.reshape(n_models, max_depth, -1), axis=(0, 2))
    q1s = np.percentile(all_results.reshape(n_models, max_depth, -1), 25, axis=(0, 2))
    q3s = np.percentile(all_results.reshape(n_models, max_depth, -1), 75, axis=(0, 2))

    return medians, q1s, q3s


if __name__ == "__main__":
    # Run experiment
    max_depth = 50
    depths = list(range(1, max_depth, 5))

    # Get results for different architecture
    medians_linear, q1s_linear, q3s_linear = measure_directional_robustness(
        max_depth, n_models=25, use_sigmoid=False, rank_fraction=1.0
    )
    # medians_nonlinear, q1s_nonlinear, q3s_nonlinear = measure_directional_robustness(
    #     max_depth, n_models=25, use_sigmoid=True, rank_fraction=1.0
    # )
    # medians_redundant, q1s_redundant, q3s_redundant = measure_directional_robustness(
    #     max_depth, n_models=25, use_sigmoid=False, rank_fraction=0.35
    # )  # Reduced rank

    # Select plotting depths
    plot_indices = [i - 1 for i in depths]
    medians_linear = [medians_linear[i] for i in plot_indices]
    q1s_linear = [q1s_linear[i] for i in plot_indices]
    q3s_linear = [q3s_linear[i] for i in plot_indices]
    # medians_nonlinear = [medians_nonlinear[i] for i in plot_indices]
    # q1s_nonlinear = [q1s_nonlinear[i] for i in plot_indices]
    # q3s_nonlinear = [q3s_nonlinear[i] for i in plot_indices]
    # medians_redundant = [medians_redundant[i] for i in plot_indices]
    # q1s_redundant = [q1s_redundant[i] for i in plot_indices]
    # q3s_redundant = [q3s_redundant[i] for i in plot_indices]

    # Set clean plotting style
    fig, ax = plt.subplots(figsize=(5, 3))

    # Plot linear case
    ax.plot(depths, medians_linear, label="Linear", linewidth=2)
    ax.fill_between(depths, q1s_linear, q3s_linear, alpha=0.2)

    # # Plot nonlinear case
    # ax.plot(depths, medians_nonlinear, label="Linear + sigmoid", linewidth=2)
    # ax.fill_between(depths, q1s_nonlinear, q3s_nonlinear, alpha=0.2)

    # # Plot redundant case
    # ax.plot(depths, medians_redundant, label="Linear + reduced rank", linewidth=2)
    # ax.fill_between(depths, q1s_redundant, q3s_redundant, alpha=0.2)

    # Clean up the plot
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    # Set labels and title
    ax.set_xlabel("Layer Depth", fontsize=15)
    ax.set_ylabel("Sensitivity to \n random perturbation", fontsize=15)

    # Add legend with clean style
    # ax.legend(frameon=False, fontsize=12)
    ax.tick_params(axis="both", which="major", labelsize=12)

    # Adjust layout
    plt.tight_layout()
    plt.savefig(
        "output/sensitivity_to_random_input_perturbation.pdf", format="pdf", dpi=300
    )
    plt.show()
