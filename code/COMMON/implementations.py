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

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    
    # number of samples in total
    num_row = y.shape[0]
    
    # number of samples per fold
    interval = int(num_row / k_fold)
    
    # set the seed
    np.random.seed(seed) 
    
    # get indices
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    
    print("Number of samples in total: ", y.shape[0])
    print("Number of folds: ",k_fold, " of", interval, "samples.")
    
    return np.array(k_indices)

def cross_validation(y, x, k_indices, k, lambda_, degree):
    """Return the training and testing losses (rmse) for RIDGE REGRESSION. The training is done on the kth subfold. 
        (x,y) : input data used for the regression.
        k     : kth subgroup to test (others are used for the training) 
        lambd_a : parameter for the ridge regression
        degree : degree of the basis polynomial fonction used for the regression."""
    
    # get k'th subgroup in test, others in train
    x_te = x[k_indices[k,:]]
    y_te = y[k_indices[k,:]]
    x_tr = x[np.union1d(k_indices[:k,:], k_indices[k+1:,:])]
    y_tr = y[np.union1d(k_indices[:k,:], k_indices[k+1:,:])]

    # build data with polynomial degree
    phi_te = build_poly(x_te, degree)
    phi_tr = build_poly(x_tr, degree)
        
    # ridge regression
    w_tr = ridge_regression(y_tr, phi_tr, lambda_)
    
    # calculate the loss for train and test data
    rmse_tr = np.sqrt(2*compute_mse(y_tr, phi_tr, w_tr))
    rmse_te = np.sqrt(2*compute_mse(y_te, phi_te, w_tr))

    return rmse_tr, rmse_te

def cross_validation_visualization(lambds, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    
    plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")
    
def cross_validation_lambda(degree, lambda_min, lambda_max, lambda_steps, k_fold, seed_split_data):
    """ Given a degree for the regression it finds the optimal lambda in the log interval [lambda_min, lambda_max]
    thanks to cross   validation on 'k_folds' different training/testing sets. """
        
    # tested lambdas
    lambdas = np.logspace(lambda_min, lambda_max, lambda_steps)
    
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed_split_data)
    
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
 
    for  lambda_ in lambdas:

        rmse_tr_all = []
        rmse_te_all = []
            
        for k in range(k_fold):        
        
            # compute losses for the k'th fold
            rmse_tr_tmp, rmse_te_tmp = cross_validation(y, x, k_indices, k, lambda_, degree)

            # store losses 
            rmse_tr_all.append(rmse_tr_tmp)
            rmse_te_all.append(rmse_te_tmp)
        
        # store mean losses
        rmse_tr.append(np.mean(rmse_tr_all))
        rmse_te.append(np.mean(rmse_te_all))
    
    # extract the optimal value for lambda
    lambda_opt = lambdas[rmse_te.index(min(rmse_te))]
    
    # plot the training and testing errors for the different values of lambda
    cross_validation_visualization(lambdas, rmse_tr, rmse_te)
    
    return lambda_opt

def cross_validation_degree(lambda_, degree_min, degree_max, k_fold):
    """ Given a degree for the regression it finds the optimal degree in the log interval [degree_min, degree_max]
    thanks to cross   validation on 'k_folds' different training/testing sets. """
        
    # tested degrees
    degrees = np.arange(degree_min, degree_max+1)
    
    # split data in k fold
    seed = 1
    k_indices = build_k_indices(y, k_fold, seed)
    
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
 
    for  degree in degrees:

        rmse_tr_all = []
        rmse_te_all = []
            
        for k in range(k_fold):        
        
            # compute losses for the k'th fold
            rmse_tr_tmp, rmse_te_tmp = cross_validation(y, x, k_indices, k, lambda_, degree)

            # store losses 
            rmse_tr_all.append(rmse_tr_tmp)
            rmse_te_all.append(rmse_te_tmp)
        
        # store mean losses
        rmse_tr.append(np.mean(rmse_tr_all))
        rmse_te.append(np.mean(rmse_te_all))
    
    # extract the optimal degree between 'degree_min' and 'degree_max'
    degree_opt = degrees[rmse_te.index(min(rmse_te))]
    
    # plot the training and testing errors for the different degrees
    cross_validation_visualization(degrees, rmse_tr, rmse_te)
    
    return degree_opt