import numpy as np
import matplotlib.pyplot as plt
from build_poly import build_poly
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

def cross_validation_ridge_regression(y, x, k_indices, k, lambda_, degree):
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

    # build data with polynomial degree
    phi_te = build_poly(x_te, degree)
    phi_tr = build_poly(x_tr, degree)
        
    # ridge regression
    w_tr, loss_tr = ridge_regression(y_tr, phi_tr, lambda_)
    
    # calculate the loss for train and test data
    rmse_tr = np.sqrt(2*compute_mse(y_tr, phi_tr, w_tr))
    rmse_te = np.sqrt(2*compute_mse(y_te, phi_te, w_tr))

    return rmse_tr, rmse_te
    
def cross_validation_lambda_ridge_regression(y, x, degree, lambda_min, lambda_max, lambda_steps, k_fold, seed_split_data):
    """ Given a degree for the regression it finds the optimal lambda in the log interval [lambda_min, lambda_max]
    thanks to cross   validation on 'k_folds' different training/testing sets. """
        
    # tested lambdas
    lambdas = np.logspace(lambda_min, lambda_max, lambda_steps)
    
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed_split_data)
    
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
 
    for  lambda_ in lambdas:

        rmse_tr_all = []
        rmse_te_all = []
            
        for k in range(k_fold):        
        
            # compute losses for the k'th fold
            rmse_tr_tmp, rmse_te_tmp = cross_validation_ridge_regression(y, x, k_indices, k, lambda_, degree)

            # store losses 
            rmse_tr_all.append(rmse_tr_tmp)
            rmse_te_all.append(rmse_te_tmp)
        
        # store mean losses
        rmse_tr.append(np.mean(rmse_tr_all))
        rmse_te.append(np.mean(rmse_te_all))
    
    # extract the optimal value for lambda
    lambda_opt = lambdas[rmse_te.index(min(rmse_te))]
    
    # plot the training and testing errors for the different values of lambda
    cross_validation_visualization_lambda(lambdas, rmse_tr, rmse_te)
    
    return lambda_opt, rmse_tr, rmse_te

def cross_validation_degree_ridge_regression(y, x, lambda_, degree_min, degree_max, k_fold):
    """ Given a degree for the regression it finds the optimal degree in the log interval [degree_min, degree_max]
    thanks to cross   validation on 'k_folds' different training/testing sets. """
        
    # tested degrees
    degrees = np.arange(degree_min, degree_max+1)
    
    # split data in k fold
    seed = 1
    k_indices = build_k_indices(y, k_fold, seed)
    
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
 
    for  degree in degrees:

        rmse_tr_all = []
        rmse_te_all = []
            
        for k in range(k_fold):        
        
            # compute losses for the k'th fold
            rmse_tr_tmp, rmse_te_tmp = cross_validation_ridge_regression(y, x, k_indices, k, lambda_, degree)

            # store losses 
            rmse_tr_all.append(rmse_tr_tmp)
            rmse_te_all.append(rmse_te_tmp)
        
        # store mean losses
        rmse_tr.append(np.mean(rmse_tr_all))
        rmse_te.append(np.mean(rmse_te_all))
    
    # extract the optimal degree between 'degree_min' and 'degree_max'
    degree_opt = degrees[rmse_te.index(min(rmse_te))]
    
    # plot the training and testing errors for the different degrees
    cross_validation_visualization_degree(degrees, rmse_tr, rmse_te)
    
    return degree_opt, rmse_tr, rmse_te

def cross_validation_rmse_ridge_regression(y, x, lambda_, degree, k_fold):
    
    # split data in k fold
    seed = 1
    k_indices = build_k_indices(y, k_fold, seed)
    
    # define lists to store the loss of training data and test data
    rmse_tr_all = []
    rmse_te_all = []
         
    for k in range(k_fold):        
        
        # compute losses for the k'th fold
        rmse_tr_tmp, rmse_te_tmp = cross_validation_ridge_regression(y, x, k_indices, k, lambda_, degree)

        # store losses 
        rmse_tr_all.append(rmse_tr_tmp)
        rmse_te_all.append(rmse_te_tmp)
        
    # std of the losses
    mean_rmse_tr = np.mean(rmse_tr_all)
    mean_rmse_te = np.mean(rmse_te_all)
    
    # std of the losses
    std_rmse_tr = np.std(rmse_tr_all)
    std_rmse_te = np.std(rmse_te_all)
    
    # boxplot of the training and testing rmse
    plt.figure()
    plt.boxplot(np.column_stack((np.array(rmse_tr_all), np.array(rmse_te_all))), labels=['training rmse','testing rmse'])
    
    return mean_rmse_tr, mean_rmse_te, std_rmse_tr, std_rmse_te

def cross_validation_classification_ridge_regression(y, x, lambda_, degree, k_fold):
    
    # split data in k fold
    seed = 1
    k_indices = build_k_indices(y, k_fold, seed)
    
    # classification errors obtained after prediction on the testing set
    classification_errors = []
         
    for k in range(k_fold):        
        
        # get k'th subgroup in test, others in train
        x_te = x[k_indices[k,:],:]
        y_te = y[k_indices[k,:]]
        x_tr = x[np.union1d(k_indices[:k,:], k_indices[k+1:,:]),:]
        y_tr = y[np.union1d(k_indices[:k,:], k_indices[k+1:,:])]

        # build data with polynomial degree
        phi_te = build_poly(x_te, degree)
        phi_tr = build_poly(x_tr, degree)

        # ridge regression
        w_tr, loss_tr = ridge_regression(y_tr, phi_tr, lambda_)
        
        # predict
        y_pred = predict_labels(w_tr, phi_te)

        # classification error
        classification_error = len(np.argwhere(y_te-y_pred))/len(y_te)*100

        # store
        classification_errors.append(classification_error)
        
    # mean and std
    mean_classification_error = np.mean(classification_errors)
    std_classification_error = np.std(classification_errors)
    
    return mean_classification_error, std_classification_error