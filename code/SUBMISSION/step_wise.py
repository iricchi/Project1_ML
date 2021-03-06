import numpy as np 
import matplotlib.pyplot as plt
from proj1_helpers import load_csv_data, predict_labels 
from implementations_enhanced import *
from outliers import handle_outliers
from standard import standardize
from split_data import split_data
from optimize_hyperparams import *
from cross_validation import *
from costs import *


def results_r2_stepwise(list_r2_adj, indices_features):
    
    for i in range(len(list_r2_adj)):
        print('step', i+1, ': R2 adjusted =', list_r2_adj[i])
        
    print("-------------------------------------------------------")
    print("Number of features chosen:", len(indices_features))
    print("Indices of features chosen: ", indices_features)

    
def stepwise(model, R2_method, all_candidates, features, y_true, cv):
    
    """
    Stepwise function takes in input the model dictionary and the R2_method, which we have fixed
    with McFadden (Pseudo R2). 
    The algorithm does a forward elimination: starting with no variables in the model, it tests 
    the addition of each variable using a chosen model fit criterion, adding the variable (if any)
    whose inclusion gives the most statistically significant improvement of the fit, and repeating 
    this process until none improves the model to a statistically significant extent.    
    """
    # data set sizes
    numSamples = all_candidates.shape[0]
    numFeat = all_candidates.shape[1]
    
    # offset
    H = np.ones((numSamples,1)) 

    # initialization (only with the offset: lack of info)
    X = H
    # k = number of features chosen, used to comput R2 adjusted
    k = 0
    
    if cv == 0:

        # fit with the offset (no cross validation)
        if model['method'] == 'ls':

            w0, loss = least_squares(y_true,X)
            loglike0 = compute_loglikelihood_reg(y_true,X,w0)
            loglike0 = loglike0/numSamples

        elif model['method'] == 'rr':

            w0, loss = ridge_regression(y_true,X,0)  # initialize with lambda = 0  
            loglike0 = compute_loglikelihood_reg(y_true,X,w0)
            loglike0 = loglike0/numSamples

        elif model['method'] == 'lsgd':

            initial_w = np.ones(X.shape[1])
            w0, loss = least_squares_GD(y_true,X, initial_w, model['max_iters'], model['gamma'], model['threshold'],
                                        model['debug_mode'])
            loglike0 = compute_loglikelihood_reg(y_true, X, w0[-1])
            loglike0 = loglike0/numSamples

        elif model['method'] == 'lssgd':

            initial_w = np.ones(X.shape[1])
            w0, loss = least_squares_SGD(y_true,X, initial_w, model['max_iters'], model['gamma'], model['batch_size'],
                                         model['threshold'], model['debug_mode'])
            loglike0 = compute_loglikelihood_reg(y_true, X, w0[-1])
            loglike0 = loglike0/numSamples

        elif model['method'] == 'lr':

            initial_w = np.ones(X.shape[1])
            w0, loss = logistic_regression(y_true,X, initial_w, model['max_iters'], model['gamma'], model['method_minimization'],
                                           model['threshold'], model['debug_mode'])
            loglike0 = compute_loglikelihood_reg(y_true, X, w0[-1])
            loglike0 = loglike0/numSamples

        elif model['method'] == 'lrr':

            initial_w = np.ones(X.shape[1])
            w0, loss = reg_logistic_regression(y_true,X, initial_w, model['max_iters'], model['gamma'], model['method_minimization'],
                                            model['lambda_'], model['threshold'], model['debug_mode'])
            loglike0 = compute_loglikelihood_reg(y_true, X, w0[-1])
            loglike0 = loglike0/numSamples

        else:
            print('No correct type of model specified')

    elif cv == 1:
        
        # fit with the offset (with cross validation)
        model['initial_w'] = np.ones(X.shape[1])
        w_tr_tot, loss_tr_tot, loss_te_tot, success_rate  = cross_validation(y_true, X, model)
        loglike0 = np.mean(loss_te_tot)
        loglike0 = loglike0/(numSamples/model['k_fold'])
        
    else:
        
        print('No cross validation specified: cv = 0 or 1') 
        
    #fix the R2adj_max
    R2 = 0             # definition : R2 = 1 - loglike0/loglike0 = 1 - 1
    R2adj_0 = R2               
    R2adj_max = R2adj_0
    ind_max = 0  # best feature index
    del(X)
    idx_features = []
    best_R2adj = []

    for j in range(numFeat):
        
        R2_adj = []
        
        for i in range(all_candidates.shape[1]):
            # increase the features 
            X = np.concatenate((H,all_candidates[:,i].reshape(numSamples,1)), axis=1)
            k = X.shape[1] - 1   # the ones column (offset) is not considered
            
            if cv == 0:
                
                # # estimate the model error (=loglikelihood) when training on the whole dataset
                if model['method'] == 'ls':
                    
                    ws,_ = least_squares(y_true,X)
                    
                elif model['method'] == 'rr':
                    
                    ws,_ = ridge_regression(y_true,X,model['lambda_'])
                    
                elif model['method']== 'lsgd':
                    
                    initial_w = np.ones(X.shape[1])
                    ws_tot,_ = least_squares_GD(y_true,X, initial_w, model['max_iters'], model['gamma'], model['threshold'],
                                                model['debug_mode'])
                    ws = ws_tot[-1]

                elif model['method'] == 'lssgd':

                    initial_w = np.ones(X.shape[1])
                    ws_tot,_ = least_squares_SGD(y_true,X, initial_w, model['max_iters'], model['gamma'], model['batch_size'],
                                                model['threshold'], model['debug_mode'])
                    ws = ws_tot[-1]
                    
                elif model['method'] == 'lr':

                    initial_w = np.ones(X.shape[1])
                    ws_tot,_ = logistic_regression(y_true,X, initial_w, model['max_iters'], model['gamma'],
                                                   model['method_minimization'], model['threshold'], model['debug_mode'])
                    ws = ws_tot[-1]

                elif model['method'] == 'lrr':

                    initial_w = np.ones(X.shape[1])
                    ws_tot,_ = reg_logistic_regression(y_true,X, initial_w, model['max_iters'], model['gamma'],
                                                       model['method_minimization'], model['lambda_'], model['threshold'],
                                                       model['debug_mode'])
                    ws = ws_tot[-1]
                    
                else:
                    
                    print('No correct type of model specified')
                    
                # compute R-squared McFadden
                loglike = compute_loglikelihood_reg(y_true,X,ws)
                loglike = loglike/numSamples
                R2 = 1-(loglike/loglike0)

            elif cv == 1:
                
                # estimate the model error (=loglikelihood) with cross validation
                model['initial_w'] = np.ones(X.shape[1])
                w_tr_tot, loss_tr_tot, loss_te_tot, success_rate  = cross_validation(y_true, X, model)
                loss = np.mean(loss_te_tot)
                loss = loss/(numSamples/model['k_fold'])
                
                # compute R-squared McFadden 
                R2 = 1-(loss/loglike0)
                
            else:
                
                print('No cross validation specified: cv = 0 or 1')
                
            # correction depending on the number of features and of samples   
            R2_adj.append(R2 - (k/(numSamples-k-1)*(1-R2)))
                
        # take the best R2   
        R2adj_chosen = np.max(R2_adj)
        best_R2adj.append(R2adj_chosen)
        idx_chosen = np.argmax(R2_adj)
        
        if R2adj_chosen > R2adj_max:
            
            # update
            R2adj_max = R2adj_chosen
            ind_max = idx_chosen
            
            # realloc of H with the regressor chosen so that X will be build with the new H and another potential candidate
            H = np.concatenate((H, all_candidates[:,ind_max].reshape(numSamples,1)), axis = 1)
            all_candidates = np.delete(all_candidates,ind_max,1)
            
            print('--------------------------------------------------------------------------------------------')
            print('Feature chosen: ', features[ind_max][1], '(index :', features[ind_max][0], ') |', ' R2adj = ', R2adj_chosen)
            
            idx_features.append(features[ind_max][0])
            
            #deleting the feature chosen in order not to have the combination with the same features
            del(features[ind_max])
            del(X)

        else:
            break
            
    return best_R2adj, idx_features