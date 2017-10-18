def compute_gradient(y, tx, w):
    """Compute the gradient."""

    # number of samples
    N = len(y)

    # compute the vector of the errors
    e = y-tx.dot(w)

    # compute the gradient
    grad = -(1/N)*tx.T.dot(e)

    return grad