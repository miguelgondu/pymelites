'''
This script creates a class for a generic MAP_Elites implementation.

See:
    - Illuminating search spaces by mapping elites.
    https://arxiv.org/abs/1504.04909
    - The original pymap-elites implementation by J.B. Mouret et al.
    https://gitlab.inria.fr/resibots/public/py_map_elites
'''
import itertools
from pathlib import Path
import json

import numpy as np

from scipy.spatial import KDTree
from sklearn.cluster import KMeans
from operator import itemgetter

class Cell:
    '''
    This cell object maintains the current elite in the cell,
    its genotypical description in self.solution, its feature
    description in self.features and its features in self.features.
    It also maintains a certain amount of elites in each cell.

    Cells are usually indexed by their centroid. The centroid acts as
    an identifier in the MAP_Elites self.cells object.
    '''
    def __init__(self, centroid, amount_of_elites=3, metadata={}, goal=None):
        self.centroid = centroid
        self.solution = None
        self.features = None
        self.performance = None
        self.amount_of_elites = amount_of_elites
        self.elites = {}
        self.metadata = metadata
        self.goal = goal

    def _objective(self, p):
        if self.goal is None:
            return p
        else:
            return - np.abs(p - self.goal)

    def add_to_elites(self, genotype, performance):
        """
        This function adds a genotype to a set of elites by
        checking if it has a better performance than the ones
        currently in the archive.
        """
        genotype = tuple_to_string(tuple(genotype))
        if len(self.elites) < self.amount_of_elites:
            self.elites[genotype] = performance
        else:
            sorted_items = list(self.elites.items())
            sorted_items += [(genotype, self._objective(performance))]
            sorted_items.sort(key=itemgetter(1), reverse=True)
            self.elites = dict(sorted_items[:self.amount_of_elites])

    def to_dict(self):
        """
        This function returns a serialized version of
        this cell. This dict has the following keys:

        {
            "centroid": cell's centroid (tuple),
            "solution": best performing genotype (you define its type),
            "features": solution's features (tuple or dict),
            "performance": solution's performance (number-like),
            "elites": an archive of the best elites (dict),
            "metadata": metadata passed by simulate (yours to define),
            "goal": the goal (performance aiming for this)
        }
        """
        # TODO: Once the generic functions for serializing
        # have been implemented, add them here.
        if self.solution is not None:
            if isinstance(self.solution, (list, np.ndarray)):
                solution = tuple(self.solution)
            elif isinstance(self.solution, dict):
                solution = self.solution
        else:
            solution = None

        if self.features is not None:
            if isinstance(self.features, (list, np.ndarray)):
                features = tuple(self.features)
            elif isinstance(self.features, dict):
                features = self.features
        else:
            features = None

        document = {
            "centroid": tuple(self.centroid),
            "solution": solution,
            "features": features,
            "performance": self.performance,
            "elites": self.elites,
            "metadata": self.metadata,
            "goal": self.goal
        }

        return document
    
    @classmethod
    def from_dict(cls, cell_doc, amount_of_elites=3):
        cell = cls(cell_doc["centroid"], amount_of_elites)
        cell.solution = cell_doc["solution"]
        cell.features = cell_doc["features"]
        cell.performance = cell_doc["performance"]
        cell.elites = cell_doc["elites"]
        cell.metadata = cell_doc["metadata"]
        cell.goal = cell_doc["goal"]
        return cell


class MAP_Elites:
    def __init__(self, random_solution, random_selection, random_variation, simulate, goal=None):
        '''
        The initialization of a MAP_Elites object. It takes as input
        - random_solution() returns a random "genotype".
        - random_selection(X) grabs an element at random from a set
          of solutions.
        - random_variation(x) mutates the genotype x.
        - simulate(x) grabs a genotype x and simulates it, recording a
          low-dimensional feature description and a performance. It
          should return either a tuple (performance(x), features(x)) or
          a tuple (performance(x), features(x), metadata(x)). This
          metadata will be stored in the cell for the best performing
          solution (genotype).
        - goal (optional) specifices a target performance to aim to.
          (in some experiments, we don't want to maximize performance,
          but rather aim for a performance that is close to a goal). If
          this goal is None, the goal turns to maximizing performance.

          features(x) should return a dict {feature: value} with the
          same features as the partition specified while creating cells.
        '''
        self.random_solution = random_solution
        self.random_selection = random_selection
        self.random_variation = random_variation
        self.simulate = simulate

        if goal is not None:
            assert isinstance(goal, (float, int)), "Goal must be float or int."
        self.goal = goal

        self.partition = None
        self.cells = None
        self.centroids = None
        self.centroids_tree = None
        self.solutions = {}

    def create_cells(self, partition, amount_of_elites=3):
        '''
        This function creates the cells for a grid partition of the
        low-dimensional feature space. These will be used to maintain
        the archive once `compute_archive` is used.

        It takes:
        - partition: a description of how to divide the feature
          space, it's a dictionary {
                "feat_1": (lower_bound, upper_bound, amount_of_cells),
                ...
                "feat_n": (lower_bound, upper_bound, amount_of_cells)
          }, with this partition the cells will be created in a gridlike
          fashion, and each one of them identified by their centroid. The
          features will be ordered alphabetically when performing computations.
        - amount_of_elites: an integer stating how many elites are going to
          be kept for each cell. Each cell can keep multiple elites (which
          can be useful when you want to store a certain amount of
          high-performing genotypes in each cell).

        Once it is run, the following attributes are stored in the class:
        - self.cells: a dict {centroid: Cell object}.
        - self.centroids: an array with the centroids of all cells.
        - self.centroids_tree: a KDTree built with the centroids for fast
            closest neighbor querying.
        '''
        midpoints = {}
        for feature, tuple_ in partition.items():
            a, b, n = tuple_
            h = (b - a)/(n - 1)
            midpoints[feature] = np.linspace(a + (1/2)*h, b - (1/2)*h, n - 1)
        midpoints_items = list(midpoints.items())
        midpoints_items.sort(key=itemgetter(0))
        midpoint_arrays = [item[1] for item in midpoints_items]
        centroids = itertools.product(*midpoint_arrays)
        cells = {centroid: Cell(centroid, amount_of_elites, goal=self.goal) for centroid in centroids}
        self.cells = cells
        self.centroids = np.array(list(self.cells.keys()))
        self.centroids_tree = KDTree(self.centroids)

    def create_cells_CVT(self, partition, amount_of_elites, samples=25000):
        '''
        This function creates the cells for a CVT partition of the
        feature space.

        It takes:
          - partition: a description of how to divide the feature
              space, it's a list [
                    (lower_bound_f1, upper_bound_f1, amount_of_cells_f1),
                    ...
                    (lower_bound_fn, upper_bound_fn, amount_of_cells_fn)
                ], with this partition the cells will be created, each 
                one of them identified by their centroid.
          - amount_of_elites: an integer stating how many elites are going
            to be kept for each cell. Each cell can keep multiple elites
            (which can be useful when you want to store a certain amount
            of high-performing genotypes in each cell).
        '''

        # Sampling random points and running kmeans
        amount_of_cells = max([tuple_[2] for tuple_ in partition])
        feature_space_dim = len(partition)
        # print(f"Sampling {samples} for the CVT creation.")
        random_samples = np.zeros((samples, feature_space_dim))
        for i, tuple_ in enumerate(partition):
            random_samples[:, i] = np.random.uniform(tuple_[0], tuple_[1], samples)

        # print(f"Finding the centroids using KMeans.")
        self.centroids = KMeans(n_clusters=amount_of_cells).fit(random_samples).cluster_centers_
        # print(f"Computing the KDTree.")
        self.centroids_tree = KDTree(self.centroids)

        # Creating the cell objects
        # TODO: the tuple stuff is dumb, I should find a better way.
        cells = {
            tuple(centroid): Cell(tuple(centroid), amount_of_elites) for centroid in self.centroids
        }
        self.cells = cells
        # print(f"Cells successfully created.")

    def _get_tuple_from_feature_dict(self, b):
        """
        Transforms dict into tuple by considering the keys
        in alphabetical order.
        """
        keys = list(b.keys())
        keys.sort()
        return tuple([b[key] for key in keys])

    def get_cell(self, b):
        """
        Gets the cell corresponding to features b.
        Takes a dict or a array-like.
        """
        if isinstance(b, dict):
            # Transform it to a tuple-like.
            b_tuple = self._get_tuple_from_feature_dict(b)
        else:
            b_tuple = b

        # Queries the KDTree for the closest centroid in
        # the grid defined by the partition, and returns
        # the respective cell.
        centroid_pos = self.centroids_tree.query(b_tuple)[1]
        centroid = self.centroids_tree.data[centroid_pos]
        return self.cells[tuple(centroid)]

    def _objective(self, r):
        """
        Returns r if there's no goal, otherwise it returns
        the distance to the goal times -1 (because we're maximizing).
        """
        if self.goal is None:
            return r
        else:
            return - np.abs(r - self.goal)

    def process_solution(self, x_prime, metadata={}):
        '''
        In the main body of the MAP-Elites algorithm, a "simulation"
        is performed to get both the feature description of the
        genotype x_prime and the performance p_prime. Once these are
        computed, the archive is maintained by adding the solution
        to the respective cell if they are the new elite.
        '''
        _tuple = self.simulate(x_prime)
        if len(_tuple) == 2:
            p_prime, b_prime = _tuple
        elif len(_tuple) == 3:
            # Player has passed metadata for the cell
            p_prime, b_prime, metadata = _tuple
        else:
            raise RuntimeError("The simulate function should return (p, features) or (p, features, metadata).")

        if p_prime is None:
            print("Performance was none. Ignoring this simulation.")
            return

        cell = self.get_cell(b_prime)
        # Update the cell's attributes &
        # Maintain the list of current solutions
        obj_prime = self._objective(p_prime)
        if cell.solution is None or self._objective(cell.performance) < obj_prime:
            cell.solution = x_prime
            cell.performance = p_prime
            cell.features = b_prime
            cell.metadata = metadata
            self.solutions[cell.centroid] = x_prime

        cell.add_to_elites(x_prime, p_prime)

    def compute_archive(self, generations, iterations_per_gen, initial_iterations=None, generation_path='.', save_each_gen=True, comment="", verbose=True):
        '''
        This function computes the archive, which is stored
        in the self.cells object.

        Each cell maintains the best performenace, the best
        solution x (also called genotype), the feature
        description of this best genotype and also metadata.

        This computation follows (loosely) the notation
        in https://arxiv.org/abs/1504.04909.

        Input:
        - generations: number of generations (int).
        - iterations_per_gen: number of iterations
          per generation (int).
        - initial_iterations: number of iterations
          in the initial generation (the one that
          creates the first random genotypes).
        - generation_path: path in which
          the generation files will be saved. (str)
        - save_each_gen: whether or not to save
          each generation. (bool)
        - comment: string that will be included
          in the name of the generations. (str)

        TODO:
            - change the amount of zeros in the saving of the file dynamically given 'generations'
            - Implement parallelization (what's the best way around this?)
            - make the generation path OS-independent.
        '''
        if self.cells is None:
            # TODO: this should be better worded, or better implemented.
            raise ValueError("Cells is None. Please run create_cells or create_cells_CVT first.")
    
        generation_path = Path(generation_path)
        for g in range(generations):
            if verbose:
                print(f"="*80)
                print(f"Generation: {g}")

            # Initialization loops
            if initial_iterations is None:
                initial_iterations = iterations_per_gen
            if g == 0:
                for it in range(initial_iterations):
                    # We get a random genotype.
                    if verbose:
                        # print(f"-"*80)
                        print(f"Iteration: {it}", end="\r", flush=True)
                    while True:
                        try:
                            x_prime = self.random_solution()
                            break
                        except (ValueError, AssertionError):
                            print("WARNING: There is something wrong with the random solution creator.")
                            pass

                    # We compute the descriptors and add them
                    # to the relevant cell.
                    # try:
                    #     self.process_solution(x_prime)
                    # except ValueError as e:
                    #     print(f"Couldn't process genotype {x_prime}. Check your random creators and mutators.")
                    #     print(e)
                    #     continue
                    self.process_solution(x_prime)

            # Update loops
            else: # g > 0
                for it in range(iterations_per_gen):
                    # Variations to the x_prime.
                    if verbose:
                        print(f"Iteration: {it}", end="\r", flush=True)
                    x = self.random_selection(list(self.solutions.values()))

                    # Attempts to mutate a genotype 5 different times.
                    for _ in range(5):
                        try:
                            x_prime = self.random_variation(x)
                            break
                        except ValueError:
                            print(f"WARNING: Couldn't mutate genotype {x}. Check your random creators and mutators.")
                    else:
                        # i.e. couldn't mutate x_prime, so the code
                        # continues the computation of the archive.
                        continue

                    # and again we compute the descriptors and add
                    # the solution to the relevant cell
                    try:
                        self.process_solution(x_prime)
                    except ValueError:
                        print(f"WARNING: Couldn't process genotype {x_prime}. Check your random creators and mutators.")
                        continue

            if save_each_gen:
                self.write_cells(
                    generation_path / f"generation_{comment}_{g:09d}.json"
                )

            if g == generations - 1 and not save_each_gen:
                self.write_cells(
                    generation_path / f"generation_{comment}_{g:09d}.json"
                )

    def write_cells(self, path):
        '''
        This function writes what's on the self.cells
        object into readable JSON.
        '''
        document = {
            tuple_to_string(centroid): cell.to_dict() for centroid, cell in self.cells.items()
        }
        # print(document)
        with open(path, "w") as file_:
            json.dump(document, file_)

def tuple_to_string(tuple_):
    '''
    This function takes a tuple and converts it to a string.
    This string is JSON parsable.
    '''
    return str(list(tuple_))
