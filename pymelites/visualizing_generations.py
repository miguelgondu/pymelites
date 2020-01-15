import json
import glob
import matplotlib.pyplot as plt
import numpy as np

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

def _plot_generation(filepath, xlims=None, ylims=None, vmin=None, vmax=None):
    with open(filepath) as fp:
        generation = json.load(fp)
    
    points = np.zeros((len(generation), 2))
    colors = np.zeros(len(generation))
    # How to deal with None's?
    i = 0
    for doc in generation.values():
        if doc["performance"] is None:
            np.delete(points, i, axis=0)
            np.delete(colors, i)
            # points[i, :] = doc["centroid"]
            # colors[i] = 0
            # i += 1
        else:
            points[i, :] = doc["centroid"]
            colors[i] = doc["performance"]
            i += 1
    
    _, ax = plt.subplots(1, 1, figsize=(10, 10))
    scatter = ax.scatter(points[:, 0], points[:, 1], c=colors, vmin=vmin, vmax=vmax)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    plt.colorbar(scatter)

    title = get_name_from_path(filepath).replace("_", " ")
    ax.set_title(title)
    plt.savefig(filepath.replace(".json", ".jpg"), format="jpg")
    # plt.show()
    plt.close()

def plot_generations(filepaths, xlims, ylims):
    """
    This function takes an iterable with the paths of
    the generation_{d}.json outputted by the MAP_Elites
    compute_archive function, and creates an image for each
    generation file.

    Input:
        - filepaths
        - xlims
        - ylims
    Output:
        None, but it creates images.
    """
    filepaths = list(glob.glob('./generation_*.json'))
    vmin, vmax = get_plot_params(filepaths)

    for i, filepath in enumerate(filepaths):
        print(f"{i+1}/{len(filepaths)}")
        _plot_generation(
            filepath, xlims=xlims, ylims=ylims,
            vmin=vmin, vmax=vmax
        )
