import numpy as np
import matplotlib.pyplot as plt

def cross_validation_visualization(lambds, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    
    plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")
    
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

def cross_validation(y, x, k_indices, k, lambda_, degree):
    """Return the training and testing losses (rmse) for RIDGE REGRESSION. The training is done on the kth subfold. 
        (x,y) : input data used for the regression.
        k     : kth subgroup to test (others are used for the training) 
        lambd_a : parameter for the ridge regression
        degree : degree of the basis polynomial fonction used for the regression."""
    
    # get k'th subgroup in test, others in train
    x_te = x[k_indices[k,:]]
    y_te = y[k_indices[k,:]]
    x_tr = x[np.union1d(k_indices[:k,:], k_indices[k+1:,:])]
    y_tr = y[np.union1d(k_indices[:k,:], k_indices[k+1:,:])]

    # build data with polynomial degree
    phi_te = build_poly(x_te, degree)
    phi_tr = build_poly(x_tr, degree)
        
    # ridge regression
    w_tr = ridge_regression(y_tr, phi_tr, lambda_)
    
    # calculate the loss for train and test data
    rmse_tr = np.sqrt(2*compute_mse(y_tr, phi_tr, w_tr))
    rmse_te = np.sqrt(2*compute_mse(y_te, phi_te, w_tr))

    return rmse_tr, rmse_te
    
def cross_validation_lambda(y, x, degree, lambda_min, lambda_max, lambda_steps, k_fold, seed_split_data):
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
            rmse_tr_tmp, rmse_te_tmp = cross_validation(y, x, k_indices, k, lambda_, degree)

            # store losses 
            rmse_tr_all.append(rmse_tr_tmp)
            rmse_te_all.append(rmse_te_tmp)
        
        # store mean losses
        rmse_tr.append(np.mean(rmse_tr_all))
        rmse_te.append(np.mean(rmse_te_all))
    
    # extract the optimal value for lambda
    lambda_opt = lambdas[rmse_te.index(min(rmse_te))]
    
    # plot the training and testing errors for the different values of lambda
    cross_validation_visualization(lambdas, rmse_tr, rmse_te)
    
    return lambda_opt

def cross_validation_degree(y, x, lambda_, degree_min, degree_max, k_fold):
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
            rmse_tr_tmp, rmse_te_tmp = cross_validation(y, x, k_indices, k, lambda_, degree)

            # store losses 
            rmse_tr_all.append(rmse_tr_tmp)
            rmse_te_all.append(rmse_te_tmp)
        
        # store mean losses
        rmse_tr.append(np.mean(rmse_tr_all))
        rmse_te.append(np.mean(rmse_te_all))
    
    # extract the optimal degree between 'degree_min' and 'degree_max'
    degree_opt = degrees[rmse_te.index(min(rmse_te))]
    
    # plot the training and testing errors for the different degrees
    cross_validation_visualization(degrees, rmse_tr, rmse_te)
    
    return degree_opt