import numpy as np

def sigmoid(z):
    """apply sigmoid function on z."""
    
    return np.exp(z)/(np.exp(z)+1)