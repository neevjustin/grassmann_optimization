import numpy as np


def orthonormalize(Q):
    """QR-based retraction to stay on the Grassmann manifold."""
    Q_new, R = np.linalg.qr(Q)
    return Q_new


def tangent_projection(Q, G):
    """
    Project Euclidean gradient G onto the tangent space at Q on Gr(k,n).
    """
    return G - Q @ (Q.T @ G)


def grassmann_step(Q, grad_fn, step_size):
    """
    Perform one Riemannian gradient step on the Grassmann manifold.
    
    Parameters:
        Q: current orthonormal basis (n x k)
        grad_fn: function returning Euclidean gradient at Q
        step_size: learning rate
    
    Returns:
        Updated orthonormal Q
    """
    G = grad_fn(Q)
    G_tangent = tangent_projection(Q, G)
    Q_new = Q - step_size * G_tangent
    return orthonormalize(Q_new)
