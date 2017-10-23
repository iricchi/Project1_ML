# -*- coding: utf-8 -*-
"""a function of ploting figures."""
import numpy as np
import matplotlib.pyplot as plt


def cross_validation_visualization_lambda(lambds, loss_tr, loss_te):
    """visualization the curves of loss_tr and loss_te."""
    plt.semilogx(lambds, loss_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, loss_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("loss")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation_lambda")

def cross_validation_visualization_degree(degrees, loss_tr, loss_te):
    """visualization the curves of loss_tr and loss_te."""
    plt.plot(degrees, loss_tr, marker=".", color='b', label='train error')
    plt.plot(degrees, loss_te, marker=".", color='r', label='test error')
    plt.xlabel("degree")
    plt.ylabel("loss")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation_degree")
    
def cross_validation_visualization_gamma(gammas, loss_tr, loss_te):
    """visualization the curves of loss_tr and loss_te."""
    plt.plot(gammas, loss_tr, marker=".", color='b', label='train error')
    plt.plot(gammas, loss_te, marker=".", color='r', label='test error')
    plt.xlabel("gamma")
    plt.ylabel("loss")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation_gamma")
    
def bias_variance_decomposition_visualization(degrees, loss_tr, loss_te):
    """visualize the bias variance decomposition."""
    loss_tr_mean = np.expand_dims(np.mean(loss_tr, axis=0), axis=0)
    loss_te_mean = np.expand_dims(np.mean(loss_te, axis=0), axis=0)
    plt.plot(
        degrees,
        loss_tr.T,
        'b',
        linestyle="-",
        color=([0.7, 0.7, 1]),
        label='train',
        linewidth=0.3)
    plt.plot(
        degrees,
        loss_te.T,
        'r',
        linestyle="-",
        color=[1, 0.7, 0.7],
        label='test',
        linewidth=0.3)
    plt.plot(
        degrees,
        loss_tr_mean.T,
        'b',
        linestyle="-",
        label='train',
        linewidth=3)
    plt.plot(
        degrees,
        loss_te_mean.T,
        'r',
        linestyle="-",
        label='test',
        linewidth=3)
    plt.ylim(0.2, 0.7)
    plt.xlabel("degree")
    plt.ylabel("error")
    plt.title("Bias-Variance Decomposition")
    plt.savefig("bias_variance")
