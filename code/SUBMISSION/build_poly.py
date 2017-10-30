import numpy as np

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    
    X = np.vander((x[:,0]).T, degree+1, increasing=True)
    
    for i in range(1,np.shape(x)[1],1):
        feat = (x[:,i]).T
        vander = np.vander(feat, degree+1, increasing=True)
        #remove the column of 1 at the beginning of each vander
        vander = np.delete(vander, 0,axis = 1)
        #concatenation
        X = np.concatenate((X, vander), axis=1)
    
    return X