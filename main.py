import marimo

__generated_with = "0.12.7"
app = marimo.App(width="columns", auto_download=["ipynb"])


@app.cell
def _():
    import marimo as mo

    import evolution
    import plotting
    import helper
    return evolution, helper, mo, plotting


@app.cell
def _(evolution):
    # Evolution simulation
    layer_stats = evolution.parallel_run_evolution(
        32 * 4,
        n_generations=200,
        population_size=100,
        mutation_std=0.1,
        mutation_rate=0.2,
        eval_fraction=0.2,
        max_depth=1,
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
def _(layer_stats, plotting):
    plotting.plot_stacked_ancestry_grid(
        layer_stats["ancestry_proportions"][:6], figsize=(15, 20), save=True
    )
    return


@app.cell
def _(evolution):
    list_of_max_depth = [1, 3, 5, 7, 9, 11, 13, 15]

    fitness = {}
    mean_optimality = {}

    for max_depth in list_of_max_depth:
        fitness[str(max_depth)] = {}
        mean_optimality[str(max_depth)] = {}
        res = evolution.parallel_run_evolution(
            32 * 200,
            n_generations=80,
            population_size=20,
            mutation_std=0.1,
            mutation_rate=1,
            eval_fraction=1,
            max_depth=max_depth,
            dim=10,
            normalize=True,
            use_sigmoid=False,
        )
        fitness[str(max_depth)] = res["fitness"]
        mean_optimality[str(max_depth)] = res["mean"]
    return fitness, list_of_max_depth, max_depth, mean_optimality, res


@app.cell
def _(mean_optimality, plotting):
    plotting.plot_fitness_grid(fitness_dict=mean_optimality, figsize=(10, 5))
    return


@app.cell
def _(mean_optimality, plotting):
    plotting.plot_fitness_grid_by_population(mean_optimality, figsize=(10, 5))
    return


@app.cell
def _(helper, mean_optimality):
    df = helper.get_half_max_indices(mean_optimality)
    df
    return (df,)


@app.cell
def _(df, plotting):
    plotting.plot_half_max_heatmap(df)
    return


@app.cell
def _(fitness, plotting):
    plotting.plot_median_fitness_by_generation(fitness, save=False)
    return


@app.cell
def _(fitness, plotting):
    plotting.plot_fitness_violin_by_layer(
        fitness, generations_to_plot=[1, 3, 5, 7, 9, 11, 13, 15], save=True
    )
    return


@app.cell
def _(mean_optimality, plotting):
    plotting.plot_optimality_by_layer_at_last_generation(mean_optimality)
    return


if __name__ == "__main__":
    app.run()
