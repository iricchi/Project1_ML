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
    H = np.ones((n,1)) 

    # initialization (only with the offset: lack of info)
    X = H
    k = 0
    
    # fit with the offset
    if model['method'] == 'ls':
        
        w0, loss = least_squares(y_true,X)
        
    elif model['method'] == 'rr':
        
        w0, loss = ridge_regression(y_true,X,0)  # initialize with lambda = 0  
        
    elif model['method'] == 'lsgd':
        
        initial_w = np.ones(X.shape[1])
        w0, loss = least_squares_GD(y_true,X, initial_w, model['max_iters'], model['gamma'], model['threshold'], model['debug_mode'])
        
    elif model['method'] == 'lssgd':
        
        initial_w = np.ones(X.shape[1])
        w0, loss = least_squares_SGD(y_true,X, initial_w, model['max_iters'], model['gamma'], model['batch_size'], model['threshold'],
                                    model['debug_mode'])
        
    elif model['method'] == 'lr':
        
        initial_w = np.ones(X.shape[1])
        w0, loss = logistic_regression(y_true,X, initial_w, model['max_iters'], model['gamma'], model['method_minimization'],
                                       model['threshold'], model['debug_mode'])
        
    elif model['method'] == 'lrr':
        
        initial_w = np.ones(X.shape[1])
        w0, loss = reg_logistic_regression(y_true,X, initial_w, model['max_iters'], model['gamma'], model['method_minimization'],
                                        model['lambda_'], model['threshold'], model['debug_mode'])
        
    else:
        print('No correct type of model specified')
        

    # get the R2 of reference
    if R2_method == 'loss':
        
        sse = loss*2*numSamples
        sst = np.sum((y_true - y_true.mean())**2)
        R2 = np.abs((sst-sse)/sst)
        
    elif R2_method == 'Tjur':
        
        ind_back, ind_sig = idx_2labels(y_true, [0,1])
        y_ = X.dot(w0)
        R2 = 0
        
    elif R2_method == 'McFadden'  and model['method'] in ['ls', 'rr']:
        
        loglike0 = compute_loglikelihood_reg(y_true,X,w0)
        R2 = 0 
        
    elif R2_method == 'McFadden' and model['method'] in ['lsgd', 'lssgd', 'lr', 'lrr']:
        
        loglike0 = compute_loglikelihood_reg(y_true, X, w0[-1])
        R2 = 0
        
    else:
        print('No correct method of R2 specified')
        
        

    #fix the R2adj_max
    R2adj_0 = R2 - (k/(numSamples-k-1)*(1-R2))
    R2adj_max = R2adj_0
    ind_max = 0  # best feature index
    del(X)
    idx_features = []
    best_R2adj = []

    for j in range(numFeat):
        
        R2_adj = []
        
        for i in range(all_candidates.shape[1]):

            X = np.concatenate((H,all_candidates[:,i].reshape(numSamples,1)), axis=1)
            
            if cv == 0:
                
                if model['method'] == 'ls':
                    
                    ws , loss = least_squares(y_true,X)
                    
                elif model['method'] == 'rr':
                    
                    ws, loss = ridge_regression(y_true,X,model['lambda_'])
                    
                elif model['method']== 'lsgd':
                    
                    initial_w = np.ones(X.shape[1])
                    ws, loss = least_squares_GD(y_true,X, initial_w, model['max_iters'], model['gamma'], model['threshold'],
                                                model['debug_mode'])
                    
                elif model['method'] == 'lssgd':

                    initial_w = np.ones(X.shape[1])
                    ws, loss = least_squares_SGD(y_true,X, initial_w, model['max_iters'], model['gamma'], model['batch_size'],
                                                model['threshold'], model['debug_mode'])

                elif model['method'] == 'lr':

                    initial_w = np.ones(X.shape[1])
                    ws, loss = logistic_regression(Y,X, initial_w, model['max_iters'], model['gamma'], model['method_minimization'],
                                                   model['threshold'], model['debug_mode'])

                elif model['method'] == 'lrr':

                    initial_w = np.ones(X.shape[1])
                    ws, loss = reg_logistic_regression(Y,X, initial_w, model['max_iters'], model['gamma'],
                                                       model['method_minimization'], model['lambda_'], model['threshold'],
                                                       model['debug_mode'])
                else:
                    print('No correct type of model specified')
                    
            elif cv == 1:
                
                if model['method'] == 'ls':
                    
                    w_tr_tot, loss_tr_tot, loss_te_tot = cross_validation(y_true, X, model)
                    ws = w_tr_tot[np.argmin(loss_te_tot)]
                    loss = np.min(loss_te_tot)
                    
                elif model['method'] in ['rr', 'lsgd', 'lssgd', 'lr', 'lrr']:
                    
                    ws, loss_tr, loss, lambda_opt = optimize_lambda(Y, X, model['lambda_min'], model['lambda_max'],
                                                                    model['lambda_steps'], model)
                    
                else:
                    print('No correct type of model specified')    
                    
            else:
                print('No cross validation specified: cv = 0 or 1')
                
            if R2_method == 'loss':
                
                SSE = loss*2*numSamples
                SST = np.sum((y_true- y_true.mean())**2)
                R2 = np.abs((SST-SSE)/SST) 
                
            elif R2_method == 'Tjur':
                
                ind_back, ind_sig = idx_2labels(y_true, [0,1])
                if len(ind_sig) == 0 or len(ind_back) ==0:
                    print('No signal detected')
                    R2_adj.append(0)
                else: 
                    y_ = X.dot(ws)
                    R2 = np.abs((np.mean(y_[ind_sig]) - np.mean(y_[ind_back])))
            
            elif R2_method == 'McFadden'  and model['method'] in ['ls', 'rr']:
                
                loglike = compute_loglikelihood_reg(y_true,X,ws)
                R2 = 1-(loglike/loglike0)
                
            elif R2_method == 'McFadden' and model['method'] in ['lsgd', 'lssgd', 'lr', 'lrr']:
                
                loglike = compute_loglikelihood_reg(y_true,X,ws[-1])
                R2 = 1-(loglike/loglike0)
                
            else:
                print('No correct method of R2 specified') 
                
            k = len(ws)-1 
            R2_adj.append(R2 - (k/(numSamples-k-1)*(1-R2)))         
            
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