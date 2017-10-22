from sigmoid import sigmoid
import numpy as np

def compute_gradient_mse(y, tx, w):
    """Compute the gradient of the MSE."""

    # number of samples
    N = len(y)

    # compute the vector of the errors
    e = y-tx.dot(w)

    # compute the gradient
    grad = -(1/N)*tx.T.dot(e)

    return grad

def compute_gradient_logLikelihood_reg(y, tx, w, lambda_=0):
    """compute the gradient of the likelihood for logistic regression.
    'lambda_' = 0 for the normal likelihood.
    'lambda_' > 0 for the penalized likelihood. """

    grad = tx.T.dot(sigmoid(tx.dot(w))-y) + 2*lambda_*w
    
    return grad

def compute_hessian_logLikelihood_reg(y, tx, w, lambda_=0):
    """compute the Hessian of the likelihood for logistic regression. 
    'lambda_' = 0 for the normal likelihood.
    'lambda_' > 0 for the penalized likelihood. """
    
    # diagonal matrix S
    S = np.zeros((y.shape[0],y.shape[0]))
    for i in range(1, y.shape[0]):
        S[i,i] = sigmoid(tx[i,:].T.dot(w))*(1-sigmoid(tx[i,:].T.dot(w)))
    
    # hessian
    hess = tx.T.dot(S).dot(tx) + 2*lambda_
    
    return hess