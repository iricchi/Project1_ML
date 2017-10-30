# -*- coding: utf-8 -*-
import numpy as np
from sigmoid import sigmoid
"""Functions used to compute the loss."""

def compute_mse_reg(y, tx, w, lambda_=0):
    """Calculate the cost which is the mean squared error (mse), regularized when lambda_>0."""
   
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
    """Calculate the cost which is the mean absolute error (mae), regularized when lambda_>0."""
   
    # number of samples
    N = len(y)
    
    # compute the error
    e = y-tx.dot(w)
    
    # mean absolute error
    mae = abs(e).mean()
    
    # regularized 
    mae_reg = mae + lambda_*w.T.dot(w)
    
    return mae_reg


def compute_loglikelihood_reg(y, tx, w, lambda_=0):
    """Caluculate the cost  which is the log likelihood, regularized when lambda_>0 """
    
    # log-likelihood 
    loglikelihood = np.sum(np.log(1+np.exp(tx.dot(w))) - y*(tx.dot(w))) + lambda_*w.T.dot(w)
    
    return loglikelihood
    