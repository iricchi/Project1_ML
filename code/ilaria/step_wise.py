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


def results_r2_stepwise(list_r2_adj,indices_features):
    print("R2 asjusted values:")
    
    for i in range(len(list_r2_adj)):
        print(list_r2_adj[i])
    print("-------------------------------------------------------")
    print("Number of features chosen:", len(indices_features))
    print("\n")
    print("Indices of features chosen: ", indices_features)


def stepwise(model, R2_method, all_candidates, features, Y, cv):
    
    n = all_candidates.shape[0] #needed for the R^2 adjusted
    num = all_candidates.shape[1]
    H = np.ones((n,1)) #offset

    #Initialization only with offsets (lack of info)
    X = H
    k = 0 #needed for the R^2 adjusted

    if model['method'] == 'ls':
        w0, loss = least_squares(Y,X)  # The loss cannot be used as a measure for the feature selection because it's a 
        y = predict_labels(w0, X)
    elif model['method'] == 'rr':
        w0, loss = ridge_regression(Y,X,0)  # start with lambda = 0  
        y = predict_labels(w0, X)
    elif model['method'] == 'lr':
        initial_w = np.ones(X.shape[1])
        w0, loss = logistic_regression(Y,X, initial_w, model['max_iters'], model['gamma'], model['method_minimization'])
        y = predict_labels(w0[-1], X)        
    else:
        print('No correct type of model specified')
        

    if R2_method == 'loss':
        sse = loss*2*n
        sst = np.sum((Y - Y.mean())**2)  #lack of information
        R2 = np.abs((sst-sse)/sst)
    elif R2_method == 'Tjur':
        ind_back, ind_sig = idx_2labels(y, [0,1])
        y_ = X.dot(w0)
        R2 = 0
    elif R2_method == 'McFadden':
        loglike0 = compute_loglikelihood_reg(y,X,w0) #np.sum(np.log(1+np.exp(X.dot(w0))) - y*(X.dot(w0)))
        R2 = 0 # definition = 1 - loglike0/loglike0 = 1 -1       
    else:
        print('No correct method of R2 specified')
        
        

    #fix the R2adj_max
    R2adj_0 = R2 - (k/(n-k-1)*(1-R2))
    R2adj_max = R2adj_0
    ind_max = 0  # this index will show us which is the best feature chosen
    del(X)
    idx_features = []
    best_R2adj = []

    for j in range(num):
        R2_adj = []
        for i in range(all_candidates.shape[1]):

            X = np.concatenate((H,all_candidates[:,i].reshape(n,1)), axis=1)
            if cv == 0:
                if model['method'] == 'ls':
                    ws , loss = least_squares(Y,X)
                    y = predict_labels(ws, X)
                elif model['method'] == 'rr':
                    ws, loss = ridge_regression(Y,X,0)  # start with lambda = 0  
                    y = predict_labels(ws, X)
                elif model['method'] == 'lr':
                    initial_w = np.ones(X.shape[1])
                    ws, loss = logistic_regression(Y,X, initial_w, model['max_iters'], model['gamma'], model['method'])
                    y = predict_labels(w0[-1], X)
                else:
                    print('No correct type of model specified')
            elif cv == 1:
                if model['method'] == 'ls':
                    w_tr_tot, loss_tr_tot, loss_te_tot = cross_validation(Y, X, arg_ls)
                    ws = w_tr_tot[np.argmin(loss_te_tot)]
                    loss = np.min(loss_te_tot)
                    y = predict_labels(ws, X)
                elif model['method'] == 'rr':
                    ws, loss_tr, loss, lambda_opt = optimize_lambda(Y, X, lambda_min, lambda_max, lambda_steps, model)
                    y = predict_labels(ws, X)
                elif model['method'] == 'lr':
                    ws, loss_tr, loss, lambda_opt = optimize_lambda(Y, X, lambda_min, lambda_max, lambda_steps, model)
                    y = predict_labels(ws, X)
                else:
                    print('No correct type of model specified')    
                     
                    
            else:
                print('No cross validation specified: cv = 0 or 1')
                
                
            if R2_method == 'loss':
                SSE = loss*2*n
                SST = np.sum((Y- Y.mean())**2)
                R2 = np.abs((SST-SSE)/SST)   
            elif R2_method == 'Tjur':
                ind_back, ind_sig = idx_2labels(y, [0,1])
        
                if len(ind_sig) == 0 or len(ind_back) ==0:
                    print('No signal detected')
                    R2_adj.append(0)
                else: 
                    y_ = X.dot(ws)
                    R2 = np.abs((np.mean(y_[ind_sig]) - np.mean(y_[ind_back])))
            
            elif R2_method == 'McFadden':
                loglike = compute_loglikelihood_reg(y,X,ws) #np.sum(np.log(1+np.exp(X.dot(ws))) - y*(X.dot(ws)))
                R2 = 1-(loglike/loglike0)
            
            else:
                print('No correct method of R2 specified') 
            k = len(ws) -1 # k is the number of regressor I use -> -1 because I don't consider the offset
            R2_adj.append(R2 - (k/(n-k-1)*(1-R2)))         
            
        R2adj_chosen = np.max(R2_adj)
        best_R2adj.append(R2adj_chosen)
        idx_chosen = np.argmax(R2_adj)

        if R2adj_chosen > R2adj_max:
            R2adj_max = R2adj_chosen
            ind_max = idx_chosen

            H = np.concatenate((H, all_candidates[:,ind_max].reshape(n,1)), axis = 1)

            all_candidates = np.delete(all_candidates,ind_max,1)
            print('-------------------------------------------------')
            print('Feature chosen: ', features[ind_max][1], '(index :', features[ind_max][0], ')')
            idx_features.append(features[ind_max][0])
            del(features[ind_max])
            del(X)

        else:
            break
            
            
    return best_R2adj, idx_features
            
            
            
            
            

    
    

    

