import numpy as np


def least_squares(y, tx):
    """calculate the least squares."""
    # ***************************************************
    return (np.linalg.inv((tx.T).dot(tx))).dot(tx.T).dot(y)
    # ***************************************************
    raise NotImplementedError
