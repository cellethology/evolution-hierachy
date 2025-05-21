import marimo

__generated_with = "0.12.7"
app = marimo.App(width="columns", auto_download=["ipynb"])


@app.cell
def _():
    import marimo as mo

    import evolution
    import plotting
    return evolution, mo, plotting


@app.cell
def _(evolution):
    # Evolution simulation
    layer_stats = evolution.parallel_run_evolution(
        10,
        n_generations=1000,
        population_size=1000,
        mutation_std=0.0,
        mutation_rate=0.1,
        eval_fraction=0.,
        max_depth=1,
        dim=100,
        normalize=False,
        use_sigmoid=False,
    )
    return (layer_stats,)


@app.cell
def _(layer_stats, plotting):
    plotting.plot_layer_evolution(layer_stats, save=False)
    return


@app.cell
def _(layer_stats, plotting):
    plotting.plot_stacked_ancestry_grid(
        layer_stats["ancestry_proportions"][:18], figsize=(15, 20), save=True
    )
    return


@app.cell
def _(evolution):
    list_of_max_depth = [1, 5, 9]
    fitness = {}
    mean_optimality = {}
    for max_depth in list_of_max_depth:
        res = evolution.parallel_run_evolution(
            64 * 10,
            n_generations=500,
            population_size=1000,
            mutation_std=0.08,
            mutation_rate=1,
            eval_fraction=1,
            max_depth=max_depth,
            dim=10,
            normalize=False,
            use_sigmoid=False,
        )
        fitness[str(max_depth)] = res["fitness"]
        mean_optimality[str(max_depth)] = res["mean"]
    return fitness, list_of_max_depth, max_depth, mean_optimality, res


@app.cell
def _():
    # plotting.plot_fitness_violin_by_layer(
    #     fitness, generations_to_plot=[0, 3, 5, 10], save=True
    # )
    return


@app.cell
def _(fitness, plotting):
    plotting.plot_median_fitness_by_generation(fitness, save=True)
    return


@app.cell
def _(mean_optimality, plotting):
    plotting.plot_optimality_by_layer_at_last_generation(mean_optimality)
    return


if __name__ == "__main__":
    app.run()
