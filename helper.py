import numpy as np
import pandas as pd


def get_half_max_indices(fitness_dict):
    """
    Convert fitness dictionary to DataFrame showing first index where value exceeds half maximum.

    Parameters
    ----------
    fitness_dict : dict
        Nested dictionary where outer keys are max_depth (str), inner keys are population_size (str),
        and values are numpy arrays of fitness values.

    Returns
    -------
    pd.DataFrame
        DataFrame with max_depth as index, population_size as columns, and values are the first
        indices where fitness exceeds half maximum. Order of rows and columns matches input dictionary.
    """
    # Get population sizes in original order from first max_depth entry
    first_max_depth = next(iter(fitness_dict))
    population_sizes = list(fitness_dict[first_max_depth].keys())

    # Initialize DataFrame with original order of keys
    df = pd.DataFrame(index=list(fitness_dict.keys()), columns=population_sizes)

    # For each max_depth and population_size combination
    for max_depth in fitness_dict:
        for pop_size in population_sizes:
            # Get the fitness array
            fitness_array = fitness_dict[max_depth][pop_size][-1]

            # Calculate half maximum
            half_max = np.max(fitness_array) / 2

            # Find first index where value exceeds half maximum
            # If no such index exists, return length of array
            indices = np.where(fitness_array > half_max)[0]
            first_index = indices[0] if len(indices) > 0 else len(fitness_array)

            df.loc[max_depth, pop_size] = first_index

    return df
