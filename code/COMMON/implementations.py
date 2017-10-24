import numpy as np
from costs import *
from compute_gradient import *
from proj1_helpers import batch_iter

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

    # optimization loop
    for n_iter in range(max_iters):

        # update w
        w = w_tot[-1] - gamma*compute_gradient_mse(y, tx, w_tot[-1])

        # get new loss
        loss = compute_mse_reg(y, tx, w)        

        # store w and loss
        w_tot.append(w)
        loss_tot.append(loss)

    print("Gradient Descent({bi}/{ti}): loss MSE={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))

    return w_tot, loss_tot
    
def least_squares_SGD(y, tx, initial_w, max_iters, gamma, batch_size):
    """ Stochastic gradient descent algorithm. """
    
    # initialization
    w_tot = [initial_w]
    loss_tot = []
    
    # optimization loop
    for n_iter in range(max_iters):
                
        # pick randomly samples
        batches = batch_iter(y, tx, batch_size, num_batches=1, shuffle=True)

        for samples in batches:

            # read samples
            y_tmp = samples[0]
            tx_tmp = samples[1]
        
            # update w
            w = w_tot[-1] - gamma*compute_gradient_mse(y_tmp, tx_tmp, w_tot[-1])
            
            # get new loss
            loss = compute_mse_reg(y_tmp, tx_tmp, w)        
        
        # store w and loss
        w_tot.append(w)
        loss_tot.append(loss)
        
    print("Stochastic Gradient Descent({bi}/{ti}): loss MSE={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))

    return w_tot, loss_tot

def logistic_regression(y, tx, initial_w, max_iters, gamma, method):
    """ Minimization of the likelihood through gradient descent (method = 'gd' or Newton method (method = 'newton'). """
    
    # initialization
    w_tot = [initial_w]
    loss_tot = []
    
    # optimization loop
    for n_iter in range(max_iters):
        
        if method == 'gd':

            # compute the gradient 
            grad = compute_gradient_logLikelihood_reg(y, tx, w_tot[-1])

            # update w
            w = w_tot[-1] - gamma*grad

        elif method == 'newton':

            # compute the gradient and the hessian
            grad = compute_gradient_logLikelihood_reg(y, tx, w_tot[-1])
            hess = compute_hessian_logLikelihood_reg(y, tx, w_tot[-1])

            # update w
            w = np.linalg.solve(hess, hess.dot(w_tot[-1])-gamma*grad)

        else:
            print('The variable method has to be "gradient_descent" or "newton".')
            break

        # get new loss
        loss = compute_loglikelihood_reg(y, tx, w)  # Ilaria: I have changed with my version

        # store w and loss
        w_tot.append(w)
        loss_tot.append(loss)       
                  
    print("Logistic Regression ({bi}/{ti}): loss logLikelihood={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))
           
    return w_tot, loss_tot

def reg_logistic_regression(y, tx, initial_w, max_iters, gamma, method, lambda_):
    """ Minimization of the regularized likelihood through gradient descent (method = 'gd' or Newton method (method = 'newton'). """
    
    # initialization
    w_tot = [initial_w]
    loss_tot = []
    
    # optimization loop
    for n_iter in range(max_iters):
        
        if method == 'gd':

            # compute the gradient 
            grad = compute_gradient_logLikelihood_reg(y, tx, w_tot[-1], lambda_)

            # update w
            w = w_tot[-1] - gamma*grad

        elif method == 'newton':

            # compute the gradient and the hessian
            grad = compute_gradient_logLikelihood_reg(y, tx, w_tot[-1], lambda_)
            hess = compute_hessian_logLikelihood_reg(y, tx, w_tot[-1], lambda_)

            # update w
            w = np.linalg.solve(hess, hess.dot(w_tot[-1])-gamma*grad)
            
        else:
            print('The variable method has to be "gradient_descent" or "newton".')
            break

        # get new regularized loss
        loss = compute_loglikelihood_reg(y, tx, w, lambda_)
        
        # store w and loss
        w_tot.append(w)
        loss_tot.append(loss)       
                  
    print("Logistic Regression Regularized ({bi}/{ti}): loss loglikelihood={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))
           
    return w_tot, loss_tot