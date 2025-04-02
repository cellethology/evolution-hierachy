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
        24*6,
        n_generations=100,
        population_size=500,
        mutation_std=0.1,
        mutation_rate=1,
        eval_fraction=1,
        max_depth=3,
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
def _(layer_stats):
    layer_stats
    return


@app.cell
def _():
    # plotting.plot_stacked_ancestry_grid(
    #     layer_stats["ancestry_proportions"][:12],
    #     figsize=(5 * 3, 5 * 6),
    #     n_cols=2,
    #     save=False,
    # )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
