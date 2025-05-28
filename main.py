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
        32 * 10,
        n_generations=2000,
        population_size=1000,
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
    list_of_max_depth = [1, 5, 10]
    list_of_population_size = [5, 50, 500]

    fitness = {}
    mean_optimality = {}

    for max_depth in list_of_max_depth:
        fitness[str(max_depth)] = {}
        mean_optimality[str(max_depth)] = {}
        for population_size in list_of_population_size:
            res = evolution.parallel_run_evolution(
                32 * 30,
                n_generations=50,
                population_size=population_size,
                mutation_std=0.1,
                mutation_rate=1,
                eval_fraction=1,
                max_depth=max_depth,
                dim=10,
                normalize=False,
                use_sigmoid=False,
            )
            fitness[str(max_depth)][str(population_size)] = res["fitness"]
            mean_optimality[str(max_depth)][str(population_size)] = res["mean"]
    return (
        fitness,
        list_of_max_depth,
        list_of_population_size,
        max_depth,
        mean_optimality,
        population_size,
        res,
    )


@app.cell
def _(mean_optimality, plotting):
    plotting.plot_fitness_grid(fitness_dict=mean_optimality, figsize=(10,5))
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
    plotting.plot_median_fitness_by_generation(fitness["9"], save=False)
    return


@app.cell
def _(fitness, plotting):
    plotting.plot_fitness_violin_by_layer(
        fitness, generations_to_plot=[0, 3, 5, 10], save=True
    )
    return


@app.cell
def _(mean_optimality, plotting):
    plotting.plot_optimality_by_layer_at_last_generation(mean_optimality)
    return


if __name__ == "__main__":
    app.run()
