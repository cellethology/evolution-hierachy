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

    from evolution import parallel_run_evolution
    from plotting import plot_layer_evolution
    return (
        concurrent,
        expit,
        mo,
        np,
        parallel_run_evolution,
        plot_layer_evolution,
        plt,
        tqdm,
    )


@app.cell
def _(parallel_run_evolution):
    # Evolution simulation
    n_runs = 80
    layer_stats = parallel_run_evolution(
        n_runs,
        n_generations=1000,
        population_size=100,
        selection_fraction=0.9,
        mutation_std=1,
        max_depth=5,
        dim=50,
        normalize=False,
        use_sigmoid=False,
    )
    return layer_stats, n_runs


@app.cell
def _(layer_stats, plot_layer_evolution):
    plot_layer_evolution(layer_stats, save=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now, lets look how effectively each layer gets optimized for different selection/drift ratio (population_size/selection_fraction). To do this, we will fix the selection_fraction while varying the population size.""")
    return


if __name__ == "__main__":
    app.run()
