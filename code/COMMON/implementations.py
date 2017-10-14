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

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

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
        batches = batch_iter(y, tx, 1, num_batches=1, shuffle=True)

        for samples in batches:

            # read samples
            y_tmp = samples[0]
            tx_tmp = samples[1]
        
            # compute new parameters
            w = ws[-1] - gamma*compute_gradient(y_tmp, tx_tmp, ws[-1])
            
            # get new loss
            loss = compute_mse(y_tmp, tx_tmp, ws[-1])        
        
        # store w and loss
        ws.append(w)
        losses.append(loss)
        
    print("Gradient Descent({bi}/{ti}): loss MSE={l}, w0={w0}, w1={w1}".format(bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return ws, losses


