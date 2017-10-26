import numpy as np

def sigmoid(z):
    """apply sigmoid function on z."""
    
    return np.exp(z)/(np.exp(z)+1)

def predict_labels_lr(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = sigmoid(data.dot(weights))
    y_pred[np.where(y_pred <= 0.5)] = 0
    y_pred[np.where(y_pred > 0.5)] = 1
    
    return y_pred