import marimo

__generated_with = "0.10.2"
app = marimo.App(width="columns", auto_download=["ipynb"])


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from scipy.special import expit
    import concurrent.futures
    import marimo as mo

    import evolution
    import plotting
    return concurrent, evolution, expit, mo, np, plotting, plt, tqdm


@app.cell
def _(evolution):
    # Evolution simulation
    layer_stats = evolution.parallel_run_evolution(
        24 * 12,
        n_generations=100,
        population_size=500,
        mutation_std=0.1,
        mutation_rate=1,
        eval_fraction=1,
        max_depth=6,
        dim=10,
        normalize=False,
        use_sigmoid=False,
    )
    return (layer_stats,)


@app.cell
def _(layer_stats, plotting):
    plotting.plot_layer_evolution(layer_stats, save=False)
    return


@app.cell
def _(evolution):
    list_of_max_depth = [1, 3, 5, 7, 9, 11]
    fitness = {}
    for max_depth in list_of_max_depth:
        res = evolution.parallel_run_evolution(
            24 * 6,
            n_generations=70,
            population_size=1000,
            mutation_std=0.1,
            mutation_rate=1,
            eval_fraction=1,
            max_depth=max_depth,
            dim=10,
            normalize=False,
            use_sigmoid=False,
        )
        fitness[str(max_depth)] = res["fitness"]
    return fitness, list_of_max_depth, max_depth, res


@app.cell
def _(fitness, plotting):
    plotting.plot_fitness_violin_by_layer(
        fitness, generations_to_plot=[0, 3, 5, 10], save=True
    )
    return


@app.cell
def _(fitness, plotting):
    plotting.plot_median_fitness_by_generation(fitness, save=True)
    return


@app.cell
def _(evolution):
    output_morenoise = evolution.parallel_run_evolution(
        24,
        n_generations=1000,
        population_size=1000,
        mutation_std=0.1,
        mutation_rate=0.1,
        eval_fraction=1,
        max_depth=10,
        dim=10,
        normalize=False,
        use_sigmoid=False,
    )
    return (output_morenoise,)


@app.cell
def _(output_morenoise, plotting):
    plotting.plot_stacked_ancestry_grid(
        ancestry_matrices=output_morenoise["ancestry_proportions"], figsize=(8, 15)
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
