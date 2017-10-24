from plots import *
from cross_validation import *
import matplotlib.pyplot as plt
from build_poly import build_poly


###
# Compute the optimal lambda using the cross validation 
#
# @Returns:
#    - w_tot     : weights of the chosen model (the one with the best lambda)
#    - loss_tr   : minimum(!) training loss taken from the k folds of the chosen model (not sure if correct!)
#    - loss_te   : minimum(!) testing loss taken from the k folds of the chosen model
#    - lambda_opt: chosen lambda
#
###
def optimize_lambda(y, x, lambda_min, lambda_max, lambda_steps, args):
    """Optimization of the hyper-parameter lambda_ driving regularization. The best lambda_ is chosen
    as the one which gives the lowest testing loss."""
    
    # create set of lambdas to test
    lambda_set = np.logspace(lambda_min, lambda_max, lambda_steps)
    print('tested lambda_: ', lambda_set, '\n')

    # store weights
    w_list = []
    
    # store minimum losses from cross-validation from every lambda
    min_loss_tr_all = []
    min_loss_te_all = []

    for lambda_tmp in lambda_set:
        
        # update lambda_ in the model
        args['lambda_'] = lambda_tmp
        print('------------------------------------------ cross validation with lambda_ = ', lambda_tmp)

        # cross validation with lambda_tmp
        w_tr_tmp, loss_tr_tot_tmp, loss_te_tot_tmp = cross_validation(y, x, args)
        
        # store minimum losses from the k folds
        min_loss_tr_all.append(np.mean(loss_tr_tot_tmp))
        min_loss_te_all.append(np.mean(loss_te_tot_tmp))
        
        # store the weights related to the minimum loss (testing loss)
        w_list.append(w_tr_tmp[np.argmin(loss_te_tot_tmp)])
        
    print("sono arrivato")

    # extract the optimal value for lambda and the relatives weights, loss_tr, loss_te
    best_indx = min_loss_te_all.index(min(min_loss_te_all))
    
    lambda_opt = lambda_set[best_indx]
    w_tot = w_list[best_indx]
    loss_tr = min_loss_tr_all[best_indx]
    loss_te = min_loss_te_all[best_indx]
    
    
    # results
    cross_validation_visualization_lambda(lambda_set, min_loss_tr_all, min_loss_te_all)
    print('------------------------')
    print('Optimal lambda: ', lambda_opt)
    print('Associated testing loss: ', min(min_loss_te_all), '\n')

    return w_tot, loss_tr, loss_te, lambda_opt

def optimize_degree(y, x, degree_min, degree_max, degree_steps, args):
    """Optimization of the degree of the polynomial basis function. The best degree is chosen
    as the one which gives the lowest testing loss."""
    
    # tested degrees
    degree_set = np.arange(degree_min, degree_max+1, degree_steps)
    print('tested degree: ', degree_set, '\n')
    
    # store losses
    min_loss_tr_all = []
    min_loss_te_all = []

    for degree_tmp in degree_set:
        
        # update degree in the model
        args['degree'] = degree_tmp
        print('------------------------------------------ cross validation with degree = ', degree_tmp)

        # build polynomial basis function
        phi = build_poly(x, degree_tmp)
        
        # cross validation with degree_tmp
        _, loss_tr_tot_tmp, loss_te_tot_tmp = cross_validation(y, phi, args)
        
        # store
        min_loss_tr_all.append(min(loss_tr_tot_tmp))
        min_loss_te_all.append(min(loss_te_tot_tmp))

    # extract the optimal value for degree
    degree_opt = degree_set[min_loss_te_all.index(min(min_loss_te_all))]
    
    # results
    cross_validation_visualization_degree(degree_set, min_loss_tr_all, min_loss_te_all)
    print('------------------------')
    print('Optimal degree: ', degree_opt)
    print('Associated testing loss: ', min(min_loss_te_all), '\n')

    return degree_opt

def optimize_gamma(y, x, gamma_min, gamma_max, gamma_steps, args):
    """Optimization of the step gamma in descent based minimization algorithm (gradient descent 
    or newton). The best gamma is chosen as the one which gives the lowest testing loss."""
    
    # tested gamma values
    gamma_set = np.logspace(gamma_min, gamma_max, gamma_steps)
    print('tested gamma: ', gamma_set, '\n')
    
    # store losses
    min_loss_tr_all = []
    min_loss_te_all = []

    for gamma_tmp in gamma_set:
        
        # update gamma in the model
        args['gamma'] = gamma_tmp
        print('------------------------------------------ cross validation with gamma = ', gamma_tmp)
        
        # cross validation with gamma_tmp
        _, loss_tr_tot_tmp, loss_te_tot_tmp = cross_validation(y, x, args)
        
        # store
        min_loss_tr_all.append(min(loss_tr_tot_tmp))
        min_loss_te_all.append(min(loss_te_tot_tmp))

    # extract the optimal value for gamma
    gamma_opt = gamma_set[min_loss_te_all.index(min(min_loss_te_all))]
    
    # results
    cross_validation_visualization_gamma(gamma_set, min_loss_tr_all, min_loss_te_all)
    print('------------------------')
    print('Optimal gamma: ', gamma_opt)
    print('Associated testing loss: ', min(min_loss_te_all), '\n')

    return gamma_opt
