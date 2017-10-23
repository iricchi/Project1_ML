import numpy as np
import matplotlib.pyplot as plt
from plots import *
from implementations import *
from proj1_helpers import predict_labels
from costs import * 

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
    
    return np.array(k_indices)

def cross_validation_k(y, x, k_indices, k):
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

    # least square
    w_tr, loss_tr = least_squares(y_tr,x_tr)
    rmse_tr = np.sqrt(2*compute_mse_reg(y_tr,x_tr, w_tr))
    rmse_te = np.sqrt(2*compute_mse_reg(y_te,x_te, w_tr))
    
    return w_tr, rmse_tr, rmse_te


def cross_validation_ls(y,x):
    
    # define hyper-parameters
    seed = 1
    k_fold = 10
    
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    # define lists to store the loss of training data and test data
    w_tr_tot = []
    loss_tr_tot = []
    loss_te_tot = []

    for k in range(k_fold):        
        
        # k'th train and test
        w_tr_k, loss_tr_k, loss_te_k = cross_validation_k(y, x, k_indices, k)
        
        # store weights and losses
        w_tr_tot.append(w_tr_k)
        loss_tr_tot.append(loss_tr_k)                # I don't need losses in R2 stepwise
        loss_te_tot.append(loss_te_k)
        
    return w_tr_tot, loss_tr_tot, loss_te_tot

def cross_validation_r(y, x, k_indices, k, lambda_):
    x_te = x[k_indices[k,:]]
    y_te = y[k_indices[k,:]]
    x_tr = x[np.union1d(k_indices[:k,:], k_indices[k+1:,:])]
    y_tr = y[np.union1d(k_indices[:k,:], k_indices[k+1:,:])]
    
    w_tr = ridge_regression(y_tr, x_tr, lambda_)
    
   # calculate the loss for train and test data
    rmse_tr = np.sqrt(2*compute_mse_reg(y_tr, x_tr, w_tr))
    rmse_te = np.sqrt(2*compute_mse_reg(y_te, x_te, w_tr))
    
    return w_tr, rmse_tr, rmse_te

def cross_validation_rr(y,x):
    seed = 1
    k_fold = 10
    lambdas = np.logspace(-4, 0, 30)
    
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    w_tr_tot = []
    
    for  lambda_ in lambdas:
        w_tr_all = []
        rmse_tr_all = []
        rmse_te_all = []
        
            
        for k in range(k_fold):        
        
            # compute losses for the k'th fold
            w_tr_tmp, rmse_tr_tmp, rmse_te_tmp = cross_validation_r(y, x, k_indices, k, lambda_)

            # store losses
            w_tr_all.append(w_tr_tmp)
            rmse_tr_all.append(rmse_tr_tmp)
            rmse_te_all.append(rmse_te_tmp)
        
        # store mean losses
        rmse_tr.append(np.mean(rmse_tr_all))
        rmse_te.append(np.mean(rmse_te_all))
        w_tr_tot.append(np.mean(w_tr_all, axis=0))
        
    return w_tr_tot, rmse_tr, rmse_te