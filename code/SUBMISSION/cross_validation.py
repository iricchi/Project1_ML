import numpy as np
from implementations_enhanced import *
from proj1_helpers import *
from sigmoid import *


def build_k_indices(num_samples, k_fold, seed=0, debug_mode=0):
    """build k indices for k-fold."""
    
    # number of samples per fold
    interval = int(num_samples / k_fold)
    
    # set the seed
    np.random.seed(seed) 
    
    # get indices
    indices = np.random.permutation(num_samples)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    
    if debug_mode:
        print("Number of samples in total: ", num_samples)
        print("Number of folds: ",k_fold, " of", interval, "samples.")
    
    return np.array(k_indices)

def cross_validation_k(y, X, k_indices, k, args):
    
    # get k'th subgroup in test, others in train
    X_te = X[k_indices[k,:],:]
    y_te = y[k_indices[k,:]]
    X_tr = X[np.union1d(k_indices[:k,:], k_indices[k+1:,:]),:]
    y_tr = y[np.union1d(k_indices[:k,:], k_indices[k+1:,:])]
           
    # train with Least Squares
    if args['method'] == 'ls':
       
        w_tr,_ = least_squares(y_tr, X_tr)
        
    # train with Ridge Regression
    if args['method'] == 'rr':
        
        w_tr,_ = ridge_regression(y_tr, X_tr, args['lambda_'])
    
    # train with Least Squares Gradient Descent
    if args['method'] == 'lsgd':
        
        w_tr_tot,_ = least_squares_GD(y_tr, X_tr, args['initial_w'], args['max_iters'], args['gamma'], args['threshold'],
                                      args['debug_mode'])
        w_tr = w_tr_tot[-1]
    
    # train with Least Squares Stochastic Gradient Descent
    if args['method'] == 'lssgd':
        
        w_tr_tot,_ = least_squares_SGD(y_tr, X_tr, args['initial_w'], args['max_iters'], args['gamma'], args['batch_size'],
                                       args['threshold'], args['debug_mode'])
        w_tr = w_tr_tot[-1]
    
    # train with Logistic Regression
    if args['method'] == 'lr':
        
        w_tr_tot,_ = logistic_regression(y_tr, X_tr,args['initial_w'], args['max_iters'], args['gamma'], args['method_minimization'],
                                         args['threshold'], args['debug_mode'])
        w_tr = w_tr_tot[-1]

    # train with Regularized Logistic Regression
    if args['method'] == 'lrr':
       
        w_tr_tot,_ = reg_logistic_regression(y_tr, X_tr,args['initial_w'], args['max_iters'], args['gamma'],
                                             args['method_minimization'], args['lambda_'], args['threshold'], args['debug_mode'])
        w_tr = w_tr_tot[-1]
        
    # check if regularization 
    if 'lambda_' in args:
        lambda_ = args['lambda_']
    else:
        lambda_ = 0
        
    # compute losses with the training weights 
    if args['loss'] == 'rmse':

        loss_tr = np.sqrt(2*compute_mse_reg(y_tr, X_tr, w_tr, lambda_))
        loss_te = np.sqrt(2*compute_mse_reg(y_te, X_te, w_tr, lambda_))        
        
    if args['loss'] == 'mae':

        loss_tr = compute_mae_reg(y_tr, X_tr, w_tr, lambda_)
        loss_te = compute_mae_reg(y_te, X_te, w_tr, lambda_)
     
    if args['loss'] == 'loglikelihood':
        
        loss_tr = compute_loglikelihood_reg(y_tr, X_tr, w_tr, lambda_)
        loss_te = compute_loglikelihood_reg(y_te, X_te, w_tr, lambda_)
        
    #compute success rateo (=classification error)
    if args['method'] == 'lr' or args['method'] == 'lrr':
        y_model = predict_labels_log(w_tr, X_te)
    else:
        y_model = predict_labels(w_tr, X_te)
        
    pos = 0
    neg = 0
    for i in range (len(y_model)):
        if y_te[i] == y_model[i]:
            pos += 1
        else:
            neg += 1

    success_rate = pos/(pos+neg)
    
    #if args['debug_mode']:
    #    print(success_rate)
    
    return w_tr, loss_tr, loss_te, success_rate
    
def cross_validation(y, X, args, debug_mode=0):
            
    # store weights and losses
    w_tr_tot = []
    loss_tr_tot = []
    loss_te_tot = []
    success_rate_tot = []

    # build k folds of indices 
    num_samples = y.shape[0]
    k_indices = build_k_indices(num_samples, args['k_fold'], debug_mode=0)
    
    # cross validation
    for k in range(args['k_fold']):        
        
        # k'th train and test
        w_tr_k, loss_tr_k, loss_te_k, success_rate = cross_validation_k(y, X, k_indices, k, args)
        
        # store weights and losses
        w_tr_tot.append(w_tr_k)
        loss_tr_tot.append(loss_tr_k)
        loss_te_tot.append(loss_te_k)
        success_rate_tot.append(success_rate)

    return w_tr_tot, loss_tr_tot, loss_te_tot, np.mean(success_rate_tot)