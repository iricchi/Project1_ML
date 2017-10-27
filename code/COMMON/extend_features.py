import numpy as np
from standard import standardize
from build_poly import build_poly

def extend_features(X, feature_names, degree, is_add_log = False, index_log = [0,1,2,5], log_names = ['log(DER_mass_MMC)', 'log(DER_mass_MMC)','log(DER_mass_transverse_met_lep)', 'log(DER_mass_vis)', 'log(DER_mass_jet_jet)']):

    # build polynomial basis function
    poly = build_poly(X, degree)

    # create feature set
    X_ext = poly[:,1:]

    # adding degree names
    features_names_ext = []
    for i in range(len(feature_names)):
        for d in range(degree):
            features_names_ext.append(feature_names[i] + '_power_' + str(d+1))
            
    print('---------------------------')
    print('Features have been set to the power(s):', str(list(range(degree+1)[1:])))

    # adding logarithmic transformation
    if is_add_log:

        for i, indx in enumerate(index_log):

            # adding logarithmic transformation
            X_ext = np.concatenate((X_ext, np.log(1+np.abs(X[:,indx])).reshape(len(X),1)),axis =1)

            # adding logarithm name
            features_names_ext.append(log_names[i])
       
        print(str(len(index_log)), 'logarithmic features have been added.')

   # standardize the data
    X_ext,_ ,_ = standardize(X_ext) 
    
    print('Data have been standardized.')
    print('---------------------------')

    # list of feature names and indices
    features_ext = []
    for i in range(len(features_names_ext)):
        features_ext.append((i,features_names_ext[i]))
    
    return X_ext, features_ext