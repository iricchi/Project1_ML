# Standardize dataset 
import numpy as np

def standardize(x):
    """Standardize the original data set."""
    
    # get the vector of means (for each feature)
    mean_x = np.mean(x, axis = 0)
    
    # pair wise substraction of those means in each feature space
    x = x - mean_x
    
    # get the vector of standard deviations (for each feature)
    std_x = np.std(x, axis = 0)
    
    # pair wise division of those standard deviations in each feature space
    x = x / std_x
    
    return x, mean_x, std_x