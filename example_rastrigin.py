'''
In this script, I test my MAP_elites implementation on the 6D-Rastrigin.

In general, you'll need to provide the MAP_Elites object the following functions:
    self.random_solution = random_solution
    self.random_selection = random_selection
    self.random_variation = random_variation
    self.performance = performance
    self.feature_descriptor = feature_descriptor
    self.partition = partition

For the rastrigin example,
    self.random_solution = a random sample from the -2\pi, 2\pi hypercube.
    self.random_selection = random.choice I guess.
    self.random_variation = Gaussian noise.
    self.performance = the actual rastrigin function.
    self.feature_descriptor = project to the first two coordinates.
    self.partition = divide it into a particular grid.

TODO: fix this docstring
'''
import numpy as np
import random
import matplotlib.pyplot as plt

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

# print(map_elites.centroids)

# _, ax = plt.subplots(1, 1)
# ax.scatter(map_elites.centroids[:, 0], map_elites.centroids[:, 1])
# plt.show()

map_elites.compute_archive(100, 100, generation_path='.')
plot_generations("./generation_*.json", partitions=partitions)