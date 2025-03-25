import numpy as np

import concurrent.futures
from layered_system import LayeredSystem, uniform_sphere_gaussian


def run_evolution(
    n_generations=10,
    population_size=1000,
    selection_fraction=0.1,
    mutation_std=0.1,
    dim=20,
    use_sigmoid=False,
    rank_fraction=1,
    max_depth=20,
    normalize=False,
    seed=None,
):
    """Run evolutionary simulation"""
    if seed is not None:
        np.random.seed(seed)

    # Initialize layeredSystem
    system = LayeredSystem(
        dim=dim,
        max_depth=max_depth,
        rank_fraction=rank_fraction,
        use_sigmoid=use_sigmoid,
    )

    # Initialize with random population
    population = uniform_sphere_gaussian(population_size, dim=system.dim)

    # Generate optimal direction
    optimal_direction = uniform_sphere_gaussian(1, dim=system.dim)[0]

    # Track mean and standard deviation at each layer
    layer_stats = {
        "mean": np.zeros((system.max_depth + 1, n_generations)),
        "stdev": np.zeros((system.max_depth + 1, n_generations)),
    }

    # Compute optimal outputs at each layer
    optimal_outputs = system._backward_pass(optimal_direction.reshape(1, -1))

    for gen in range(n_generations):
        # Forward pass through all layers
        layer_outputs = system._forward_pass(population)

        # Track statistics at each layer
        for layer_idx, (layer_out, opt_out) in enumerate(
            zip(layer_outputs, optimal_outputs)
        ):
            angles = system._compute_angles(layer_out, opt_out[0])
            layer_stats["mean"][layer_idx, gen] = np.mean(angles)
            layer_stats["stdev"][layer_idx, gen] = np.std(angles)

        # Selection
        final_angles = system._compute_angles(layer_outputs[-1], optimal_outputs[-1])
        n_select = int(population_size * selection_fraction)
        selected_idx = np.argsort(-final_angles)[:n_select]
        selected = population[selected_idx]

        # Mutation
        n_offspring = population_size - n_select
        offspring = selected[np.random.choice(n_select, n_offspring)]
        offspring += np.random.randn(n_offspring, system.dim) * (
            mutation_std / np.sqrt(system.dim)
        )

        # Create new population
        population = np.vstack([selected, offspring])
        population = population / np.linalg.norm(population, axis=1, keepdims=True)

    # Mean standardize to first generation
    if normalize:
        factor = layer_stats["mean"][:, 0, None]
        layer_stats["mean"] /= factor
        layer_stats["stdev"] /= factor

    return layer_stats


def run_evolution_with_kwargs(kwargs):
    return run_evolution(**kwargs)


def parallel_run_evolution(n_runs, **kwargs):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                run_evolution_with_kwargs,
                {**kwargs, "seed": np.random.randint(0, 1000000)},
            )
            for _ in range(n_runs)
        ]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    # Aggregate results
    aggregated_results = {}
    for key in ["mean", "stdev"]:
        values = np.array([result[key] for result in results])
        if key == "mean":
            aggregated_results[key] = np.mean(values, axis=0)
        if key == "stdev":
            aggregated_results[key] = np.std(values, axis=0)
    return aggregated_results
