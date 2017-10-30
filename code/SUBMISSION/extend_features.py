import numpy as np
from standard import standardize
from build_poly import build_poly

def extend_features(X, feature_names, degree, is_add_log = False, index_log = [0,1,2,5], log_names = ['log(DER_mass_MMC)', 'log(DER_mass_MMC)','log(DER_mass_transverse_met_lep)', 'log(DER_mass_vis)', 'log(DER_mass_jet_jet)'], new_feat = True):

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

    # adding 16 features (momentum)
    if new_feat :
        
        features_4 = X[:,[13,16,23,26]]
        eta_angles = X[:, [14,17,24,27]]
        phi_angles = X[:,[15,18,25,28]]        
        
        P_mom_comp1 = features_4*np.cos(phi_angles)
        P_mom_comp2 = features_4*np.sin(phi_angles)
        P_mom_comp3 = features_4*np.sin(eta_angles)
        P_mom_mod = features_4*np.cosh(eta_angles)
    

        names = ['PRI_tau_pt_mom_comp1', 'PRI_lep_pt_mom_comp1', 'PRI_jet_leading_pt_mom_comp1',
                 'PRI_jet_subleading_pt_mom_comp1', 'PRI_tau_pt_mom_comp2', 'PRI_lep_pt_mom_comp2',
                 'PRI_jet_leading_pt_mom_comp2', 'PRI_jet_subleading_pt_mom_comp2', 'PRI_tau_pt_mom_comp3',
                 'PRI_lep_pt_mom_comp3', 'PRI_jet_leading_pt_mom_comp3', 'PRI_jet_subleading_pt_mom_comp3',
                 'PRI_tau_pt_mom_module', 'PRI_lep_pt_mom_module', 'PRI_jet_leading_pt_mom_module',
                 'PRI_jet_subleading_pt_mom_module']
        
        for i in range(len(names)):
            features_names_ext.append(names[i])
            
        X_ext = np.concatenate((X_ext, P_mom_comp1), axis = 1)
        X_ext = np.concatenate((X_ext, P_mom_comp2), axis = 1)
        X_ext = np.concatenate((X_ext, P_mom_comp3), axis = 1)
        X_ext = np.concatenate((X_ext, P_mom_mod), axis = 1)    
        
        print("16 Features of the momentum have been added")

        
    # adding logarithmic transformation
    if is_add_log:

        for i, indx in enumerate(index_log):

            # adding logarithmic transformation
            X_ext = np.concatenate((X_ext, np.log(1+np.abs(X[:,indx])).reshape(len(X),1)),axis =1)

            # adding logarithm name
            features_names_ext.append(log_names[i])
       
        print(str(len(index_log)), 'logarithmic features have been added.')
        
    # list of feature names and indices
    features_ext = []
    for i in range(len(features_names_ext)):
        features_ext.append((i,features_names_ext[i]))
    
    return X_ext, features_ext