"""
This script will take a generation in which the features
are given by a dict with >2 keys, and aggregate them in
lower dimensions. It will also plot them.
"""

import json

def aggregate_generation(generation, keys):
    agg_generation = {}
    all_features_keys = None
    for k, doc in generation.items():
        if doc["performance"] == None:
            continue

        if all_features_keys is None:
            all_features_keys = list(doc["features"].keys())
            all_features_keys.sort()
            keys.sort()

        # Find the reduced centroid for the keys in keys.
        indices = [
            all_features_keys.index(key) for key in keys
        ]
        centroid = [doc["centroid"][index] for index in indices]
        # index_0 = all_features_keys.index(keys[0])
        # index_1 = all_features_keys.index(keys[1])
        # centroid = [doc["centroid"][index_0], doc["centroid"][index_1]]

        # If it hasn't been stored or if the performance is better, store it
        if str(centroid) not in agg_generation:
            agg_generation[str(centroid)] = {
                **doc,
                "centroid": centroid
            }

        if doc["performance"] > agg_generation[str(centroid)]["performance"]:
            agg_generation[str(centroid)] = {
                **doc,
                "centroid": centroid
            }
    
    return agg_generation

