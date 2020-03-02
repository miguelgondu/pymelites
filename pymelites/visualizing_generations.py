import json
import glob
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter

def get_name_from_path(filepath):
    return filepath.split("/")[-1].split(".")[0]

def get_plot_params(filepaths):
    vmin, vmax = np.Inf, np.NINF
    for filepath in filepaths:
        with open(filepath) as fp:
            generation = json.load(fp)
        
        for doc in generation.values():
            if doc["performance"] is None:
                continue

            if doc["performance"] > vmax:
                vmax = doc["performance"]
            if doc["performance"] < vmin:
                vmin = doc["performance"]

    return vmin, vmax

def _plot_generation(filepath, partition=None, vmin=None, vmax=None):
    with open(filepath) as fp:
        generation = json.load(fp)

    # Getting the plotting parameters from the partition.
    partition_items = list(partition.items())
    partition_items.sort(key=itemgetter(0))

    key_x = partition_items[0][0]
    key_y = partition_items[1][0]
    xlims = partition_items[0][1][:2]
    ylims = partition_items[1][1][:2]

    points = np.zeros((len(generation), 2))
    colors = np.zeros(len(generation))

    i = 0
    for doc in generation.values():
        if doc["performance"] is None:
            np.delete(points, i, axis=0)
            np.delete(colors, i)
        else:
            if isinstance(doc["features"], (list, tuple, np.ndarray)):
                index_x, index_y = key_x, key_y
            elif isinstance(doc["features"], dict):
                # Sort the keys in doc["features"]
                keys = list(doc["features"].keys())
                keys.sort()

                # Pick the index_x and index_y according to the positions
                # of key_x and key_y in the sorted array, because the
                # centroid was built in alphabetical order.
                index_x, index_y = keys.index(key_x), keys.index(key_y)
            else:
                raise RuntimeError(f"Features is of unexpected type {type(doc['features'])}")
            points[i, :] = (doc["centroid"][index_x], doc["centroid"][index_y])
            colors[i] = doc["performance"]
            i += 1
    
    _, ax = plt.subplots(1, 1, figsize=(10, 10))
    scatter = ax.scatter(points[:, 0], points[:, 1], c=colors, vmin=vmin, vmax=vmax)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    if isinstance(partition_items[0][0], str):
        ax.set_xlabel(partition_items[0][0])
        ax.set_ylabel(partition_items[1][0])
    plt.colorbar(scatter)

    title = get_name_from_path(filepath).replace("_", " ")
    ax.set_title(title)
    plt.savefig(filepath.replace(".json", ".jpg"), format="jpg")
    # plt.show()
    plt.close()

def plot_generations(filepaths, partition=None):
    """
    This function takes an iterable with the paths of
    the generation_{d}.json outputted by the MAP_Elites
    compute_archive function, and creates an image for each
    generation file.

    Input:
        - filepaths
        - partition.
    Output:
        None, but it creates images.
    """
    files = list(glob.glob(filepaths))
    vmin, vmax = get_plot_params(files)

    for i, filepath in enumerate(files):
        print(f"{i+1}/{len(files)}")
        _plot_generation(
            filepath, partition=partition, vmin=vmin, vmax=vmax
        )
