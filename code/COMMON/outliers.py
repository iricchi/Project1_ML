import numpy as np

"""
Set delete to 1 if you want to delete -999 from the dataset,
Set delete to 0 if you want to substitude -999 with 0,
Set delete to 'mean' if you want to substitude -999 with the mean
Set delete to 'median' if you want to substitude -999 with the median
"""
def handle_outliers(input_data, yb, outlier_value, delete):
    
    X = input_data

    if delete==1:
        
        print('samples with -999 are removed from the dataset ')
        
        Y = yb
        
        for i in range(input_data.shape[1]):
            del_idx = np.where(X[:,i] == outlier_value)
            X = np.delete(X, del_idx, 0)      
            Y = np.delete(Y, del_idx, 0)
        
    elif delete==0:
        
        print('-999 are replaced by 0')
        
        # labels are unchanged
        Y = yb
        
        for i in range(X.shape[1]):
            X[np.where(X[:,i]==-999),i] = 0
        
    elif delete=='mean':
        
        print('-999 are replaced by the mean value of the feature')

        # labels are unchanged
        Y = yb

        # get the feature medians
        X_tmp = X
        for i in range(input_data.shape[1]):
            X_tmp = np.delete(X_tmp, np.where(X_tmp[:,i] == outlier_value), 0)
        means = np.mean(X_tmp, axis=0)

        for i in range(input_data.shape[1]):
            X[np.where(input_data[:,i]== outlier_value),i] = means[i]
        
    elif delete=='median':
        
        print('-999 are replaced by the median value of the feature')

        # labels are unchanged
        Y = yb 

        # get the feature medians
        X_tmp = X
        for i in range(input_data.shape[1]):
            X_tmp = np.delete(X_tmp, np.where(X_tmp[:,i] == outlier_value), 0)
        medians = np.median(X_tmp, axis=0)

        for i in range(input_data.shape[1]):
            X[np.where(input_data[:,i]== outlier_value),i] = medians[i]
                    
    else:
    
        print('data unchanged !')
        
        # labels are unchanged
        Y=yb
        
    return X, Y