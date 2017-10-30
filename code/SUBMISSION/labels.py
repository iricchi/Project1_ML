import numpy as np

def idx_2labels(vet, values):
    indices_1 = np.where(vet == values[0])[0]
    indices_2 = np.where(vet == values[1])[0]
    
    return indices_1, indices_2