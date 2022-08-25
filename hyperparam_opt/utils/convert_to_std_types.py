import numpy as np


def convert_to_std_type(value):
    if isinstance(value, np.float64):
        return float(value)
    if isinstance(value, np.int64):
        return int(value)
    return value
