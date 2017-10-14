import numpy as np
from costs import *

def least_squares(y, tx):
    """calculate the least squares."""
    w = np.linalg.solve(((tx.T).dot(tx)), (tx.T).dot(y))
    
    loss = compute_mse(y, tx, w);
    
    return w, loss


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""    
    w_rr = np.linalg.solve(tx.T.dot(tx) + lambda_*np.identity(tx.shape[1]) , (tx.T).dot(y))
    
    loss = compute_mse(y, tx, w_rr);
    
    return w_rr, loss

def compute_gradient(y, tx, w):

    """Compute the gradient."""

    # number of samples
    N = len(y)

    # compute the vector of the errors
    e = y-tx.dot(w)

    # compute the gradient
    grad = -(1/N)*tx.T.dot(e)

    return grad

def least_squares_GD(y, tx, initial_w, max_iters, gamma):

    """Gradient descent algorithm."""
    
    # define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w

    # optimization loop
    for n_iter in range(max_iters):

        # compute new parameters
        w = w - gamma*compute_gradient(y, tx, w)

        # get new loss
        loss = compute_mse(y, tx, w)        

        # store w and loss
        ws.append(w)
        losses.append(loss)

    print("Gradient Descent({bi}/{ti}): loss MSE={l}, w0={w0}, w1={w1}".format(bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return ws, losses
    
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    
    """Stochastic gradient descent algorithm."""
    
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    
    # optimization loop
    for n_iter in range(max_iters):
                
        # pick randomly 'batch_size' samples
        batches = batch_iter(y, tx, batch_size, num_batches=1, shuffle=True)
        
        for samples in batches:

            # read samples
            y = samples[0]
            tx = samples[1]
        
            # compute new parameters
            w = ws[-1] - gamma*compute_gradient(y, tx, ws[-1])
            
            # get new loss
            loss = compute_mse(y, tx, ws[-1])        
        
        # store w and loss
        ws.append(w)
        losses.append(loss)
        
    print("Gradient Descent({bi}/{ti}): loss MSE={l}, w0={w0}, w1={w1}".format(bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return ws, losses
