import marimo

__generated_with = "0.10.19"
app = marimo.App(width="medium", auto_download=["ipynb"])


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
    n_runs = 500
    layer_stats = parallel_run_evolution(
        n_runs,
        n_generations=200,
        population_size=500,
        selection_fraction=0.3,
        mutation_std=0.01,
        max_depth=3,
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
