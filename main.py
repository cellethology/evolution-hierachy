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
    n_runs = 32
    layer_stats = evolution.parallel_run_evolution(
        n_runs,
        n_generations=1000,
        population_size=1000,
        mutation_std=0.3,
        max_depth=3,
        dim=10,
        normalize=False,
        use_sigmoid=False,
    )
    return layer_stats, n_runs


@app.cell
def _(layer_stats, plotting):
    plotting.plot_layer_evolution(layer_stats, save=True)
    return


@app.cell
def _(layer_stats, plotting):
    plotting.plot_stacked_ancestry_grid(
        layer_stats["ancestry_proportions"], figsize=(15, 30), n_cols=4, save=True
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now, lets look how effectively each layer gets optimized for different selection/drift ratio (population_size/selection_fraction). To do this, we will fix the selection_fraction while varying the population size.""")
    return


if __name__ == "__main__":
    app.run()
