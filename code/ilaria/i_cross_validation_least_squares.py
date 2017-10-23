import numpy as np
import matplotlib.pyplot as plt
from plots import *
from implementations import *
from proj1_helpers import predict_labels

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    
    # number of samples in total
    num_row = y.shape[0]
    
    # number of samples per fold
    interval = int(num_row / k_fold)
    
    # set the seed
    np.random.seed(seed) 
    
    # get indices
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    
    print("Number of samples in total: ", y.shape[0])
    print("Number of folds: ",k_fold, " of", interval, "samples.")
    
    return np.array(k_indices)

def cross_validation_least_square(y, x, k_indices, k, k_fold):
    """Return the training and testing losses (rmse) for RIDGE REGRESSION. The training is done on the kth subfold. 
        (x,y) : input data used for the regression.
        k     : kth subgroup to test (others are used for the training) 
        lambd_a : parameter for the ridge regression
        degree : degree of the basis polynomial fonction used for the regression."""
    
    # get k'th subgroup in test, others in train
    x_te = x[k_indices[k,:],:]
    y_te = y[k_indices[k,:]]
    x_tr = x[np.union1d(k_indices[:k,:], k_indices[k+1:,:]),:]
    y_tr = y[np.union1d(k_indices[:k,:], k_indices[k+1:,:])]

    # ridge regression
    w_tr, loss_tr = least_square(y_tr,x_tr)
    
    for k in range(k)
    
    return w_tr, x_tr, x_te, y_tr, y_te