# pymelites

This package contains a Python implementation of [the MAP-Elites evolutionary algorithm](https://arxiv.org/pdf/1504.04909.pdf), inspired on [the original pymap_elites package](https://gitlab.inria.fr/resibots/public/py_map_elites) developed by JB Mouret and his lab.

## Installing

`pymelites` was written in Python >3.6. Clone this repo and go into it

```
git clone ...
cd pymelites
```

Install the dependencies using

```
pip install -r requirements.txt
```

and also install the package (in editor mode for debugging)

```
pip install -e .
```

## Usage

### Implementing the evolution

In order to run MAP-Elites, you first need to implement the 4 functions that guide the evolutionary process:
- `random_solution()`: a function that returns a random genotype `x` (e.g. the weights of a neural network, the encoding of a robot's gait).
- `random_variation(x)`: a function that mutates `x` to return a new genotype `x'`.
- `random_selection(X)`: a function that randomly selects a solution from a list of genotypes `X` (usually, `random.choice` works well).
- `simulate(x)`: a function that runs genotype `x` and returns a tuple `(performance, features, metadata)` (or `(performance, features)`). The metadata is optional, and would be stored alongside the elite in each cell.

### Defining the behaviors

Remember that MAP-Elites segments a behavior space into cells and stores, in each cell, highest performing genotype associated with those behaviors (e.g. the fastest tall robot, or the most difficult level with 3 enemies). In order to segment the behavior space, you will also need to pass a `partition` dictionary that is structured like this:

```python
partitions = {
    "feature_1": (lower_limit_1, upper_limit_1, n_1),
    "feature_2": (lower_limit_2, upper_limit_2, n_2),
    ...
    "feature_m": (lower_limit_m, upper_limit_m, n_m)
}
```

This `partition` is related to the way in which you implement the `simulate` function: we expect `features = (b1, ..., bm)` **in alphabetical order** according to your partitions dictionary.

### Defining the object and creating cells

After you implement the 4 functions we discussed above, and after you define the partition of your behavior space, you can run

```python
map_elites = MAP_Elites(
    random_solution=random_solution,
    random_selection=random_selection,
    random_variation=random_variation,
    simulate=simulate
)

map_elites.create_cells(
    partition=partitions
)
```

### Running the evolution

At this point, you can start the evolution by specifying the number of generations and iterations per generation:

```python
map_elites.compute_archive(10, 10000, comment="some_comment", generation_path='.')
```

This also takes a `generation_path`, the path in which your generations will be stored, and a `comment` that will go into the generation's name like this: `{generation_path}/generation_{comment}_{generation_number}.json`.

### An example

To see an example of how to implement these functions and define the behavior space, check the `example_rastrigin.py` file.

## Related research

This implementation has been used in:
- [Finding Game Levels with the Right Difficulty in a few Trials through Intelligent Trial-and-Error](https://arxiv.org/pdf/2005.07677.pdf) by myself alongside Rasmus Berg Palm, David Ha and Sebastian Risi. [Check the code here](https://github.com/miguelgondu/finding_game_levels_paper).

## Licence

This code is open source under the [MIT License](https://opensource.org/licenses/MIT).
