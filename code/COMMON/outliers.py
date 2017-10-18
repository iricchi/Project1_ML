import numpy as np

"""
Set delete to 1 if you want to delete -999 from the dataset,
Set delete to 0 if you want to substitude -999 with 0,
Set delete to anyother values if you want to substitude -999 with the mean
"""
def handle_outliers(input_data, yb, outlier_value, delete)
    X = input_data

    if delete==1:
        Y = yb
        for i in range(input_data.shape[1]):
            del_idx = np.where(X[:,i] == outlier_value)
            X = np.delete(X, del_idx, 0)      
            Y = np.delete(Y, del_idx, 0)
        
    elif delete==0:
        for i in range(X.shape[1]):
            X[np.where(X[:,i]==-999),i] = 0

        
    else:
        for i in range(input_data.shape[1]):
            X = np.delete(X, np.where(X[:,i] == outlier_value), 0)

        means = np.mean(X, axis=0)

        for i in range(input_data.shape[1]):
            input_data[np.where(input_data[:,i]== outlier_value),i] = means[i]
        X = input_data
        Y = yb
        
    return X, Y