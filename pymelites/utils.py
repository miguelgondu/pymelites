import numpy as np

def to_hashable(a):
    """
    TODO: finish implementing this function properly.
    """
    if isinstance(a, np.array) or isinstance(a, list):
        return tuple(a)
    else:
        return str(a)

def to_json_parsable(a):
    """
    TODO: finish implementing this recursion.
    """
    if isinstance(a, dict):
        return a
    if isinstance(a, np.array) or isinstance(a, list):
        return tuple(a)
    