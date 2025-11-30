import marimo

__generated_with = "0.12.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import evolution
    import helper
    import plotting
    return evolution, helper, plotting


@app.cell
def _(evolution):
    # Evolution simulation
    layer_stats = evolution.parallel_run_evolution(
        32 * 8,
        n_generations=500,
        population_size=200,
        mutation_std=0.1,
        mutation_rate=0.2,
        eval_fraction=0.2,
        max_depth=2,
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
    list_of_max_depth = [1, 2, 4, 6]

    fitness = {}
    mean_optimality = {}
    mock_fitness = {}

    for max_depth in list_of_max_depth:
        fitness[str(max_depth)] = {}
        mean_optimality[str(max_depth)] = {}
        res = evolution.parallel_run_evolution(
            32 * 20,
            n_generations=100,
            population_size=300,
            mutation_std=0.2,
            mutation_rate=1,
            eval_fraction=1,
            max_depth=max_depth,
            dim=20,
            normalize=True,
            use_sigmoid=False,
            nonlinear_fitness=False,
        )
        fitness[str(max_depth)] = res["fitness"]
        mean_optimality[str(max_depth)] = res["mean"]
    return (
        fitness,
        list_of_max_depth,
        max_depth,
        mean_optimality,
        mock_fitness,
        res,
    )


@app.cell
def _(fitness, plotting):
    def _():
        (
            layer_counts,
            generation_achieve_threshold,
        ) = plotting.plot_median_fitness_by_generation(fitness, figsize=(4.5, 3), threshold=1.5, save=True)
        return layer_counts, generation_achieve_threshold


    layer_counts, generation_achieve_threshold = _()
    return generation_achieve_threshold, layer_counts


@app.cell
def _():
    high_pop = [13, 8, 4, 3]
    low_pop = [107, 27, 7, 5]
    return high_pop, low_pop


@app.cell
def _(high_pop, layer_counts, low_pop, plotting):
    plotting.plot_convergence_rate(
        layer_counts, [high_pop, low_pop], labels=['high population', 'low population'], figsize=(4,3), save=True
    )
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
def _(mean_optimality, plotting):
    plotting.plot_optimality_by_layer_at_last_generation(mean_optimality)
    return


if __name__ == "__main__":
    app.run()
