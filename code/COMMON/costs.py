# -*- coding: utf-8 -*-
import numpy as np
from sigmoid import sigmoid
"""Functions used to compute the loss."""

def compute_mse_reg(y, tx, w, lambda_=0):
    """Calculate the cost using mean squared error (mse), regularized when lambda_>0."""
   
    # number of samples
    N = len(y)
    
    # compute the error
    e = y-tx.dot(w)
    
    # mean squared error
    mse = (1/(2*N))*e.dot(e)
    
    # regularized 
    mes_reg = mse + lambda_*w.T.dot(w)
    
    return mes_reg

def compute_mae_reg(y, tx, w, lambda_=0):
    """Calculate the cost using mean absolute error (mae), regularized when lambda_>0."""
   
    # number of samples
    N = len(y)
    
    # compute the error
    e = y-tx.dot(w)
    
    # mean absolute error
    mae = abs(e).mean()
    
    # regularized 
    mae_reg = mae + lambda_*w.T.dot(w)
    
    return mae_reg

def compute_logLikelihood_reg(ylabels, tx, w, lambda_=0):
    """Calculate the cost by negative log likelihood, regularized when lambda_>0. """

    # log-likelihood
    logLikelihood = 1
    for i in range(1,ylabels.shape[0]):
        logLikelihood = logLikelihood + ylabels[i]*np.log10(sigmoid(tx[i,:].T.dot(w))) + (1-ylabels[i])*np.log10(1-sigmoid(tx[i,:].T.dot(w)))                     

    # regularized when lambda_>0
    logLikelihood_reg = logLikelihood + lambda_*w.T.dot(w)
    
    return -logLikelihood_reg