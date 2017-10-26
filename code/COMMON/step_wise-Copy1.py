import numpy as np 
import matplotlib.pyplot as plt
from proj1_helpers import load_csv_data, predict_labels 
from implementations import *
from outliers import handle_outliers
from labels import idx_2labels
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
    
    # data set sizes
    numSamples = all_candidates.shape[0]
    numFeat = all_candidates.shape[1]
    
    # offset
    H = np.ones((numSamples,1)) 

    # initialization (only with the offset: lack of info)
    X = H
    k = 0
    
    # fit with the offset
    if model['method'] == 'ls':
        
        w0, loss = least_squares(y_true,X)
        loglike0 = compute_loglikelihood_reg(y_true,X,w0)
        R2 = 0 
        
    elif model['method'] == 'rr':
        
        w0, loss = ridge_regression(y_true,X,0)  # initialize with lambda = 0  
        loglike0 = compute_loglikelihood_reg(y_true,X,w0)
        R2 = 0 
        
    elif model['method'] == 'lsgd':
        
        initial_w = np.ones(X.shape[1])
        w0, loss = least_squares_GD(y_true,X, initial_w, model['max_iters'], model['gamma'], model['threshold'],
                                    model['debug_mode'])
        loglike0 = compute_loglikelihood_reg(y_true, X, w0[-1])
        R2 = 0
        
    elif model['method'] == 'lssgd':
        
        initial_w = np.ones(X.shape[1])
        w0, loss = least_squares_SGD(y_true,X, initial_w, model['max_iters'], model['gamma'], model['batch_size'],
                                     model['threshold'], model['debug_mode'])
        loglike0 = compute_loglikelihood_reg(y_true, X, w0[-1])
        R2 = 0
        
    elif model['method'] == 'lr':
        
        initial_w = np.ones(X.shape[1])
        w0, loss = logistic_regression(y_true,X, initial_w, model['max_iters'], model['gamma'], model['method_minimization'],
                                       model['threshold'], model['debug_mode'])
        loglike0 = compute_loglikelihood_reg(y_true, X, w0[-1])
        R2 = 0
        
    elif model['method'] == 'lrr':
        
        initial_w = np.ones(X.shape[1])
        w0, loss = reg_logistic_regression(y_true,X, initial_w, model['max_iters'], model['gamma'], model['method_minimization'],
                                        model['lambda_'], model['threshold'], model['debug_mode'])
        loglike0 = compute_loglikelihood_reg(y_true, X, w0[-1])
        R2 = 0
        
    else:
        print('No correct type of model specified')
               

    #fix the R2adj_max
    R2adj_0 = R2                # k = 0
    R2adj_max = R2adj_0
    ind_max = 0  # best feature index
    del(X)
    idx_features = []
    best_R2adj = []

    for j in range(numFeat):
        
        R2_adj = []
        
        for i in range(all_candidates.shape[1]):

            X = np.concatenate((H,all_candidates[:,i].reshape(numSamples,1)), axis=1)
            k = X.shape[1] - 1
            
            if cv == 0:
                
                # # estimate the model error (=loglikelihood) when training on the whole dataset
                if model['method'] == 'ls':
                    
                    ws,_ = least_squares(y_true,X)
                    
                elif model['method'] == 'rr':
                    
                    ws,_ = ridge_regression(y_true,X,model['lambda_'])
                    
                elif model['method']== 'lsgd':
                    
                    initial_w = np.ones(X.shape[1])
                    ws,_ = least_squares_GD(y_true,X, initial_w, model['max_iters'], model['gamma'], model['threshold'],
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
                R2 = 1-((loglike-k)/loglike0)

            elif cv == 1:
                
                # estimate the model error (=loglikelihood) with cross validation
                w_tr_tot, loss_tr_tot, loss_te_tot,  = cross_validation(y_true, X, model)
                loss = np.mean(loss_te_tot)
                k = w_tr_tot[-1].shape[0]
                
                # compute R-squared McFadden 
                R2 = 1-((loss-k)/loglike0)
                
            else:
                
                print('No cross validation specified: cv = 0 or 1')
                
                
            R2_adj.append(R2)         
            
        R2adj_chosen = np.max(R2_adj)
        best_R2adj.append(R2adj_chosen)
        idx_chosen = np.argmax(R2_adj)

        if R2adj_chosen > R2adj_max:
            
            R2adj_max = R2adj_chosen
            ind_max = idx_chosen
            H = np.concatenate((H, all_candidates[:,ind_max].reshape(numSamples,1)), axis = 1)
            all_candidates = np.delete(all_candidates,ind_max,1)
            
            print('-------------------------------------------------')
            print('Feature chosen: ', features[ind_max][1], '(index :', features[ind_max][0], ')')
            idx_features.append(features[ind_max][0])
            del(features[ind_max])
            del(X)

        else:
            break
            
    return best_R2adj, idx_features