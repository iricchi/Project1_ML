import numpy as np
#Set delete to 1 if you want to delete 999 from the dataset, if you want to substitude it with the mean of the feature put 0 or any other values

def handle_outliers(input_data, yb, outlier_value, delete)
    X = input_data

    if delete:
        Y = yb
        for i in range(input_data.shape[1]):
            del_idx = np.where(X[:,i] == outlier_value)
            X = np.delete(X, del_idx, 0)      
            Y = np.delete(Y, del_idx, 0)
        
    else:
        for i in range(input_data.shape[1]):
            X = np.delete(X, np.where(X[:,i] == outlier_value), 0)

        means = np.mean(X, axis=0)

        for i in range(input_data.shape[1]):
            input_data[np.where(input_data[:,i]== outlier_value),i] = means[i]
        X = input_data
        Y = yb
        
    return X, Y