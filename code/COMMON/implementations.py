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

def least_squares_GD(y, tx, initial_w, max_iters, gamma, threshold=1e-2, debug_mode=0):
    """ Gradient descent algorithm for minimization of the mean squared error (mse). """
    
    # initialization
    w_tot = [initial_w]
    loss_tot = []
    n_iter = 0
    continue_ = True
    
    # optimization loop
    while continue_:

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
        continue_ = n_iter < max_iters and  np.linalg.norm(grad) > threshold
            
        if debug_mode and n_iter % max_iters == 0:
        
            # norm of the grad
            print('n_iter:', n_iter, ', ||grad|| =', np.linalg.norm(grad))

            # check if convergence
            plt.plot(loss_tot)
            plt.xlabel('iteration')
            plt.ylabel('likelihood')
            plt.show()
            
    if debug_mode:
        
        # check if convergence
        print('--------------------- final iteration')
        plt.plot(loss_tot)
        plt.xlabel('iteration')
        plt.ylabel('likelihood')
        plt.show()
    return w_tot, loss_tot
    
def least_squares_SGD(y, tx, initial_w, max_iters, gamma, batch_size, threshold=1e-2, debug_mode=0):
    """ Stochastic gradient descent algorithm. """
    
    # initialization
    w_tot = [initial_w]
    loss_tot = []
    n_iter = 0
    continue_ = True
    
    # optimization loop
    while continue_:
                
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
        
        # check for stopping criteria
        n_iter = n_iter + 1
        continue_ = n_iter < max_iters and  np.linalg.norm(grad) > threshold
            
        if debug_mode and n_iter % max_iters == 0:
        
            # norm of the grad
            print('n_iter:', n_iter, ', ||grad|| =', np.linalg.norm(grad))

            # check if convergence
            plt.plot(loss_tot)
            plt.xlabel('iteration')
            plt.ylabel('likelihood')
            plt.show()
            
    if debug_mode:
        
        # check if convergence
        print('--------------------- final iteration')
        plt.plot(loss_tot)
        plt.xlabel('iteration')
        plt.ylabel('likelihood')
        plt.show()
            
    return w_tot, loss_tot

def logistic_regression(y, tx, initial_w, max_iters, gamma, method, threshold=1e-2, debug_mode=0):
    """ Minimization of the likelihood through gradient descent (method = 'gd' or Newton method (method = 'newton'). """

    # initialization
    w_tot = [initial_w]
    loss_tot = []
    n_iter = 0
    continue_ = True
    
    # optimization loop
    while continue_:
        
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
            print('The variable method has to be "gd" or "newton".')
            break

        # get new loss
        loss = compute_loglikelihood_reg(y, tx, w)  # Ilaria: I have changed with my version

        # store w and loss
        w_tot.append(w)
        loss_tot.append(loss)       
        
        # check for stopping criteria
        n_iter = n_iter + 1
        continue_ = n_iter < max_iters and  np.linalg.norm(grad) > threshold
            
        if debug_mode and n_iter % max_iters == 0:
        
            # norm of the grad
            print('n_iter:', n_iter, ', ||grad|| =', np.linalg.norm(grad))

            # check if convergence
            plt.plot(loss_tot)
            plt.xlabel('iteration')
            plt.ylabel('likelihood')
            plt.show()
            
    if debug_mode:
        
        # check if convergence
        print('--------------------- final iteration')
        plt.plot(loss_tot)
        plt.xlabel('iteration')
        plt.ylabel('likelihood')
        plt.show()
                       
    return w_tot, loss_tot

def reg_logistic_regression(y, tx, initial_w, max_iters, gamma, method, lambda_, threshold=1e-2, debug_mode=0):
    """ Minimization of the regularized likelihood through gradient descent (method = 'gd' or Newton method (method = 'newton'). """
    
    # initialization
    w_tot = [initial_w]
    loss_tot = []
    n_iter = 0
    continue_ = True
    
    # optimization loop
    while continue_:
        
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
            print('The variable method has to be "gd" or "newton".')
            break

        # get new regularized loss
        loss = compute_loglikelihood_reg(y, tx, w, lambda_)
        
        # store w and loss
        w_tot.append(w)
        loss_tot.append(loss)       
        
        # check for stopping criteria
        n_iter = n_iter + 1
        continue_ = n_iter < max_iters and  np.linalg.norm(grad) > threshold
            
        if debug_mode and n_iter % max_iters == 0:
        
            # norm of the grad
            print('n_iter:', n_iter, ', ||grad|| =', np.linalg.norm(grad))

            # check if convergence
            plt.plot(loss_tot)
            plt.xlabel('iteration')
            plt.ylabel('likelihood')
            plt.show()
            
    if debug_mode:
        
        # check if convergence
        print('--------------------- final iteration')
        plt.plot(loss_tot)
        plt.xlabel('iteration')
        plt.ylabel('likelihood')
        plt.show()
            
    return w_tot, loss_tot