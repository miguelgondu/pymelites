'''
This script creates a class for a generic MAP_Elites implementation.

See:
    - Illuminating search spaces by mapping elites. https://arxiv.org/abs/1504.04909
    - The original pymap-elites implementation by J.B. Mouret et al. 
'''
import itertools
import json
import numpy as np

from scipy.spatial import KDTree
from sklearn.cluster import KMeans
from operator import itemgetter

class Cell:
    '''
    This cell object maintains the current elite in the cell, its genotypical description
    in self.solution, its feature description in self.features and its features in
    self.features. It also maintains a certain amount of elites in each cell.

    Cells are usually indexed by their centroid. The centroid acts as an identifier
    in the MAP_Elites self.cells object.
    '''
    def __init__(self, centroid, amount_of_elites=3):
        self.centroid = centroid
        self.solution = None
        self.features = None
        self.performance = None
        self.amount_of_elites = amount_of_elites
        self.elites = {}

    def add_to_elites(self, genotype, performance):
        genotype = tuple_to_string(tuple(genotype))
        if len(self.elites) < self.amount_of_elites:
            self.elites[genotype] = performance
        else:
            sorted_items = list(self.elites.items())
            sorted_items += [(genotype, performance)]
            sorted_items.sort(key=itemgetter(1), reverse=True)
            self.elites = dict(sorted_items[:self.amount_of_elites])

    def to_dict(self):
        # TODO: Once the generic functions in utils have been implemented, add them here.
        if self.solution is not None:
            if isinstance(self.solution, list) or isinstance(self.solution, np.ndarray):
                solution = tuple(self.solution)
            elif isinstance(self.solution, dict):
                solution = self.solution
        else:
            solution = None

        if self.features is not None:
            features = tuple(self.features)
        else:
            features = None

        document = {
            "centroid": tuple(self.centroid), # This one's not really necessary
            "solution": solution,
            "features": features,
            "performance": self.performance,
            "elites": self.elites
        }

        return document
    
    @classmethod
    def from_dict(cls, cell_doc, amount_of_elites=3):
        cell = cls(cell_doc["centroid"], amount_of_elites)
        cell.solution = cell_doc["solution"]
        cell.features = cell_doc["features"]
        cell.performance = cell_doc["performance"]
        cell.elites = cell_doc["elites"]
        return cell


class MAP_Elites:
    def __init__(self, random_solution, random_selection, random_variation, simulate):
        '''
        Documentation: the initialization of a MAP_Elites object.
        - random_solution() returns a random "genotype".
        - random_selection(X) grabs an element at random from a set of solutions.
        - random_variation(x) mutates the genotype x.
        - simulate(x) grabs a genotype x and simulates it, recording a low-dimensional feature
          description and a performance. It should return a tuple (performance(x), feature_description(x)).

        TODO: 
            - expose the amount of elites to maintain. The kwargs are so many though. (!)
        '''
        self.random_solution = random_solution
        self.random_selection = random_selection
        self.random_variation = random_variation
        self.simulate = simulate
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
                fashion, and each one of them identified by their centroid.
            - amount_of_elites: an integer stating how many elites are going to
              be kept for each cell. Each cell can keep multiple elites (which
              can be useful when you want to store a certain amount of high-performing
              genotypes in each cell).

        Once it is run, the following attributes are stored in the class:
            - self.cells: a dict {centroid: Cell object}.
            - self.centroids: an array with the centroids of all cells.
            - self.centroids_tree: a KDTree built with the centroids for fast
              closest neighbor querying.

        TODO: fix this docstring.
        '''
        # Creating the midpoints
        midpoints = {}
        for feature, tuple_ in enumerate(partition):
            a, b, n = tuple_
            h = (b - a)/(n - 1)
            midpoints[feature] = np.linspace(a + (1/2)*h, b - (1/2)*h, n - 1)

        centroids = itertools.product(*midpoints.values())
        cells = {centroid: Cell(centroid, amount_of_elites) for centroid in centroids}
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
          - amount_of_elites: an integer stating how many elites are going to
            be kept for each cell. Each cell can keep multiple elites (which
            can be useful when you want to store a certain amount of high-performing
            genotypes in each cell).
        '''

        # Sampling random points and running kmeans
        # TODO: being purists, amount_of_cells can be computed in the one
        # linear search being performed below.
        amount_of_cells = max([tuple_[2] for tuple_ in partition])
        feature_space_dim = len(partition)
        print(f"Sampling {samples} for the CVT creation.")
        random_samples = np.zeros((samples, feature_space_dim))
        for i, tuple_ in enumerate(partition):
            random_samples[:, i] = np.random.uniform(tuple_[0], tuple_[1], samples)
        
        print(f"Finding the centroids using KMeans.")
        self.centroids = KMeans(n_clusters=amount_of_cells).fit(random_samples).cluster_centers_
        print(f"Computing the KDTree.")
        self.centroids_tree = KDTree(self.centroids)

        # Creating the cell objects
        # TODO: the tuple stuff is dumb, I should find a better way.
        cells = {tuple(centroid): Cell(tuple(centroid), amount_of_elites) for centroid in self.centroids}
        self.cells = cells
        print(f"Cells successfully created.")

    def get_cell(self, b):
        # print(f"b: {b}")
        # print(f"the result of the query: {self.centroids_tree.query(b)}")
        # print()
        centroid_pos = self.centroids_tree.query(b)[1]
        centroid = self.centroids_tree.data[centroid_pos]
        return self.cells[tuple(centroid)]

    def process_solution(self, x_prime):
        '''
        In the main body of the MAP-Elites algorithm, a "simulation" is performed to get
        both the feature description of the genotype x_prime and the performance p_prime. Once
        these are computed, the archive is maintained by adding the solution to the respective
        cell if they are the new elite.
        '''
        p_prime, b_prime = self.simulate(x_prime)

        cell = self.get_cell(b_prime)
        # print(f"cell: {cell}")
        # Update the cell's attributes &
        # Maintain the list of current solutions
        if cell.solution is None or cell.performance < p_prime:
            cell.solution = x_prime
            cell.performance = p_prime
            cell.features = b_prime
            self.solutions[cell.centroid] = x_prime

        cell.add_to_elites(x_prime, p_prime)

    def compute_archive(self, generations, iterations_per_gen, generation_path='.', save_each_gen=True, comment=""):
        '''
        This function computes the archive, which is stored in the self.cells object.

        Each cell maintains the best performenace, the best solution x (a.k.a genotype) and
        the feature description of this best genotype.

        This computation follows (loosely) the notation in https://arxiv.org/abs/1504.04909.

        TODO:
            - change the amount of zeros in the saving of the file dynamically given 'generations'
            - Implement parallelization
        '''
        if self.cells is None:
            # TODO: this should be better worded, or better implemented.
            raise ValueError("Cells is None. Please run create_cells or create_cells_CVT first.")
        for g in range(generations):
            print(f"Generation: {g}")
            # Initialization
            if g == 0:
                for _ in range(iterations_per_gen):
                    # We get a random genotype.
                    # TODO: this is a hack, remove it when the PCG stuff has been addressed.
                    while True:
                        try:
                            x_prime = self.random_solution()
                            break
                        except (ValueError, AssertionError):
                            print("WARNING: There is something wrong with the random solution creator.")
                            pass

                    # We compute the descriptors and add it to the relevant cell.
                    # TODO: this is a hack, remove it when the PCG stuff has been addressed.
                    try:
                        self.process_solution(x_prime)
                    except ValueError:
                        print(f"Couldn't process genotype {x_prime}. Check your random creators and mutators.")
                        continue
            # Update loops
            else:
                for _ in range(iterations_per_gen):
                    # Variations to the x_prime.
                    # TODO: the random selection can be sped up by working with a set, right?
                    x = self.random_selection(list(self.solutions.values()))
                    # TODO: this is a hack, remove it when the PCG stuff has been addressed.
                    for _ in range(5):
                        try:
                            x_prime = self.random_variation(x)
                            break
                        except ValueError:
                            print(f"Couldn't mutate genotype {x}. Check your random creators and mutators.")
                    else:
                        continue

                    # and again we compute the descriptors and add the solution to the relevant cell
                    # TODO: this is a hack, remove it when the PCG stuff has been addressed.
                    try:
                        self.process_solution(x_prime)
                    except ValueError:
                        print(f"Couldn't process genotype {x_prime}. Check your random creators and mutators.")
                        continue

            if save_each_gen:
                self.write_cells(generation_path + f"/generation_{comment}_{g:05d}.json")

            if g == generations - 1 and not save_each_gen:
                self.write_cells(generation_path + f"/generation_{comment}_{g:05d}.json")

    def write_cells(self, path):
        '''
        This function writes what's on the self.cells object into readable JSON.
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
    '''
    return str(list(tuple_))
