'''
In this script, I test my MAP_elites implementation on
the 6D-Rastrigin. See more at
https://en.wikipedia.org/wiki/Rastrigin_function

In general, you'll need to provide the MAP_Elites object
the following functions:
    - random_solution() -> x, your genotypes (which could
      be (almost) whatever you want).
    - random_selection(archive) -> x, a way of selecting
      randomly from a list. Usually random.choice will
      do the trick.
    - random_variation(x) -> x', your way of mutating
      genotypes.
    - simulate(x) -> (p, f, meta), a function that
      simulates a genotype x and returns its performance
      p, its behavioral features f and optionally a metadata
      dictionary that will be stored in the cells.

In what follows you'll see example implementations of these
functions.
'''
import numpy as np
import random

from pymelites.map_elites import MAP_Elites
from pymelites.visualizing_generations import plot_generations

DIMENSIONS = 6
A = 10

def random_solution():
    return np.random.uniform(-2*np.pi, 2*np.pi, DIMENSIONS)

def random_selection(X):
    return random.choice(X)

def random_variation(x, scale=1):
    return x + np.random.normal(0, scale, size=x.size)

def performance(x):
    # The rastrigin function, in this case.
    # Since MAP_elites solves maximizations, I consider
    # the "-rastrigin" function. Thus, the max is
    # actually 0.

    # TODO: test it to see if it's actually the Rastrigin in 2D.
    n = len(x)
    return -(A*n + np.sum(x ** 2 - A*np.cos(2*np.pi*x)))

def feature_descriptor(x):
    return x[:2]

def simulate(x):
    p = performance(x)
    features = x[:2]
    features = {
        "feature_a": features[0],
        "feature_b": features[1]
    }
    return p, features


partitions = {
    "feature_a": (-2*np.pi, 2*np.pi, 100),
    "feature_b": (-np.pi, 3*np.pi, 100)
}

map_elites = MAP_Elites(
    random_solution=random_solution,
    random_selection=random_selection,
    random_variation=random_variation,
    simulate=simulate
)

map_elites.create_cells(
    partition=partitions,
    amount_of_elites=3
)

map_elites.compute_archive(10, 10000, comment="original", generation_path='.')
# Uncomment this line to plot the generations.
# plot_generations("./generation_*.json", partitions=partitions)
