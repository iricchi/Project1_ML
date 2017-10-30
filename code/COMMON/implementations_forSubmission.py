import numpy as np
from costs import *
from compute_gradient import *
from proj1_helpers import batch_iter
import matplotlib.pyplot as plt

def least_squares(y, tx):
    """Minimization of the mean squared error."""
    
    # weights
    wls = np.linalg.solve(((tx.T).dot(tx)), (tx.T).dot(y))

    # loss
    loss = compute_mse_reg(y, tx, wls);
    
    return wls, loss

def ridge_regression(y, tx, lambda_):
    """ Minimization of the penalized mean squared error with the ridge regularization. """   
    
    # weights
    wrr = np.linalg.solve(tx.T.dot(tx) + lambda_*np.identity(tx.shape[1]), (tx.T).dot(y))
    
    # loss
    loss = compute_mse_reg(y, tx, wrr, lambda_);
    
    return wrr, loss


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """ Gradient descent algorithm for minimization of the mean squared error (mse). """
    
    # initialization
    w_tot = [initial_w]
    loss_tot = []
    n_iter = 0
    
    # optimization loop
    while n_iter < max_iters:

        # compute gradient
        grad = compute_gradient_mse(y, tx, w_tot[-1])
        
        # update w
        w = w_tot[-1] - gamma*grad

        # get new loss
        loss = compute_mse_reg(y, tx, w)        

        # store w and loss
        w_tot.append(w)
        loss_tot.append(loss)

        # check for stopping criteria
        n_iter = n_iter + 1
            
    return w_tot[-1], loss_tot[-1]
    
def least_squares_SGD(y, tx, initial_w, max_iters, gamma, batch_size):
    """ Stochastic gradient descent algorithm. """
    
    # initialization
    w_tot = [initial_w]
    loss_tot = []
    n_iter = 0
    
    # optimization loop
    while n_iter < max_iters:
                
        # pick randomly samples
        batches = batch_iter(y, tx, batch_size, num_batches=1, shuffle=True)

        for samples in batches:

            # read samples
            y_tmp = samples[0]
            tx_tmp = samples[1]
        
            # compute gradient
            grad = compute_gradient_mse(y_tmp, tx_tmp, w_tot[-1])
            
            # update w
            w = w_tot[-1] - gamma*grad
            
            # get new loss
            loss = compute_mse_reg(y_tmp, tx_tmp, w)        
        
        # store w and loss
        w_tot.append(w)
        loss_tot.append(loss)
        
        n_iter = n_iter + 1
            
    return w_tot[-1], loss_tot[-1]

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """ Minimization of the likelihood through gradient descent (method = 'gd' or Newton method (method = 'newton'). """

    # initialization
    w_tot = [initial_w]
    loss_tot = []
    n_iter = 0
     
    # optimization loop
    while n_iter < max_iters:
        

        # compute the gradient 
        grad = compute_gradient_logLikelihood_reg(y, tx, w_tot[-1])

        # update w
        w = w_tot[-1] - gamma*grad

        # get new loss
        loss = compute_loglikelihood_reg(y, tx, w)  
        
        # store w and loss
        w_tot.append(w)
        loss_tot.append(loss)       
        
        # check for stopping criteria
        n_iter = n_iter + 1
            
                       
    return w_tot[-1], loss_tot[-1]

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """ Minimization of the regularized likelihood through gradient descent (method = 'gd' or Newton method (method = 'newton'). """
    
    # initialization
    w_tot = [initial_w]
    loss_tot = []
    n_iter = 0
    
    # optimization loop
    while n_iter < max_iters:
        
        # compute the gradient 
        grad = compute_gradient_logLikelihood_reg(y, tx, w_tot[-1], lambda_)

        # update w
        w = w_tot[-1] - gamma*grad

        # get new regularized loss
        loss = compute_loglikelihood_reg(y, tx, w, lambda_)
        
        # store w and loss
        w_tot.append(w)
        loss_tot.append(loss)       
        
        n_iter = n_iter + 1
            
    return w_tot[-1], loss_tot[-1]