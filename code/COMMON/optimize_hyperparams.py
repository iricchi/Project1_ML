from plots import *
from cross_validation import *
import matplotlib.pyplot as plt
from build_poly import build_poly

def optimize_lambda(y, x, lambda_min, lambda_max, lambda_steps, args, debug_mode=0):
    """Optimization of the hyper-parameter lambda_ driving regularization. The best lambda_ is chosen
    as the one which gives the lowest testing loss."""
    
    # create set of lambdas to test
    lambda_set = np.logspace(lambda_min, lambda_max, lambda_steps)
    if debug_mode:
        print('tested lambda_: ', lambda_set, '\n')

    # store weights
    w_list = []
    
    # store mean losses from cross-validation from every lambda
    mean_loss_tr_all = []
    mean_loss_te_all = []
    success_rate_all = []

    for lambda_tmp in lambda_set:
        
        # update lambda_ in the model
        args['lambda_'] = lambda_tmp
        
        if debug_mode:
            print('------------------------------------------ cross validation with lambda_ = ', lambda_tmp)

        # cross validation with lambda_tmp
        w_tr_tmp, loss_tr_tot_tmp, loss_te_tot_tmp, success_rate = cross_validation(y, x, args)
        
        # store mean losses and success rates
        mean_loss_tr_all.append(np.mean(loss_tr_tot_tmp))
        mean_loss_te_all.append(np.mean(loss_te_tot_tmp))
        success_rate_all.append(success_rate)
        
        # store the weights related to the minimum loss (testing loss)
        w_list.append(w_tr_tmp[np.argmin(loss_te_tot_tmp)])
        
    # extract the optimal value for lambda and the relatives weights, loss_tr, loss_te
    best_indx = mean_loss_te_all.index(min(mean_loss_te_all))
    
    lambda_opt = lambda_set[best_indx]
    w_opt = w_list[best_indx]
    loss_tr = mean_loss_tr_all[best_indx]
    loss_te = mean_loss_te_all[best_indx]
    
    if debug_mode:
        cross_validation_visualization_lambda(lambda_set, mean_loss_tr_all, mean_loss_te_all)
        print('Optimal lambda: ', lambda_opt)
        print('Associated testing loss: ', loss_te, '\n')

    return w_opt, loss_tr, loss_te, lambda_opt, np.mean(success_rate)

def optimize_degree(y, x, degree_min, degree_max, degree_steps, args, debug_mode=0):
    """Optimization of the degree of the polynomial basis function. The best degree is chosen
    as the one which gives the lowest testing loss."""
    
    # tested degrees
    degree_set = np.arange(degree_min, degree_max+1, degree_steps)
    if debug_mode:
        print('tested degree: ', degree_set, '\n')
    
    # store mean losses from cross validation for each degree
    mean_loss_tr_all = []
    mean_loss_te_all = []
    success_rate_all = []

    # store weights
    w_list = []
    
    for degree_tmp in degree_set:
        
        # update degree in the model
        args['degree'] = degree_tmp
        
        if debug_mode:
            print('------------------------------------------ cross validation with degree = ', degree_tmp)

        # build polynomial basis function
        phi = build_poly(x, degree_tmp)
        
        # update initial weights 
        args['initial_w'] = np.zeros(phi.shape[1]) 
        
        # cross validation with degree_tmp
        w_tr_tmp, loss_tr_tot_tmp, loss_te_tot_tmp, success_rate = cross_validation(y, phi, args)
        
        # store mean losses
        mean_loss_tr_all.append(np.mean(loss_tr_tot_tmp))
        mean_loss_te_all.append(np.mean(loss_te_tot_tmp))
        success_rate_all.append(success_rate)

        # store the weights related to the minimum loss (testing loss)
        w_list.append(w_tr_tmp[np.argmin(loss_te_tot_tmp)])
        
    # extract the optimal value for lambda and the relatives weights, loss_tr, loss_te
    best_indx = mean_loss_te_all.index(min(mean_loss_te_all))
    
    degree_opt = degree_set[best_indx]
    w_opt = w_list[best_indx]
    loss_tr = mean_loss_tr_all[best_indx]
    loss_te = mean_loss_te_all[best_indx]
    
    if debug_mode:
        cross_validation_visualization_degree(degree_set, mean_loss_tr_all, mean_loss_te_all)
        print('Optimal degree: ', degree_opt)
        print('Associated testing loss: ', loss_te, '\n')

    return w_opt, loss_tr, loss_te, degree_opt, np.mean(success_rate)

def optimize_gamma(y, x, gamma_min, gamma_max, gamma_steps, args, debug_mode=0):
    """Optimization of the step gamma in descent based minimization algorithm (gradient descent 
    or newton). The best gamma is chosen as the one which gives the lowest testing loss."""
    
    # tested gamma values
    gamma_set = np.logspace(gamma_min, gamma_max, gamma_steps)
    if debug_mode:
        print('tested gamma: ', gamma_set, '\n')
    
    # store mean losses
    mean_loss_tr_all = []
    mean_loss_te_all = []
    success_rate_all = []

    # store weights
    w_list = []
    
    for gamma_tmp in gamma_set:
        
        # update gamma in the model
        args['gamma'] = gamma_tmp
        
        if debug_mode:
            print('------------------------------------------ cross validation with gamma = ', gamma_tmp)
        
        # cross validation with gamma_tmp
        w_tr_tmp, loss_tr_tot_tmp, loss_te_tot_tmp, success_rate = cross_validation(y, x, args)
        
        # store mean losses
        mean_loss_tr_all.append(np.mean(loss_tr_tot_tmp))
        mean_loss_te_all.append(np.mean(loss_te_tot_tmp))
        success_rate_all.append(success_rate)

        # store the weights related to the minimum loss (testing loss)
        w_list.append(w_tr_tmp[np.argmin(loss_te_tot_tmp)])
        
    # extract the optimal value for lambda and the relatives weights, loss_tr, loss_te
    best_indx = mean_loss_te_all.index(min(mean_loss_te_all))
    
    gamma_opt = gamma_set[best_indx]
    w_opt = w_list[best_indx]
    loss_tr = mean_loss_tr_all[best_indx]
    loss_te = mean_loss_te_all[best_indx]
    
    if debug_mode:
        cross_validation_visualization_gamma(gamma_set, mean_loss_tr_all, mean_loss_te_all)
        print('Optimal gamma: ', gamma_opt)
        print('Associated testing loss: ', loss_te, '\n')

    return w_opt, loss_tr, loss_te, gamma_opt, np.mean(success_rate)
