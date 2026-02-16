import numpy as np


def pca_objective(S):
    """
    Returns objective function and Euclidean gradient
    for maximizing trace(Q^T S Q).
    We minimize negative trace for gradient descent.
    """

    def loss(Q):
        return -np.trace(Q.T @ S @ Q)

    def grad(Q):
        # Euclidean gradient of -trace(Q^T S Q)
        return -2 * S @ Q

    return loss, grad
