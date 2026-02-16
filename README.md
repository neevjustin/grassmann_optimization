# Optimization on Grassmann Manifolds for Subspace-Constrained Learning

This repository implements Riemannian gradient descent on the Grassmann manifold Gr(k,n)
to solve a subspace-constrained PCA objective.

We optimize:

    max_Q Tr(Q^T S Q)

subject to Q^T Q = I

using:

- Tangent space projection
- QR-based retraction
- Intrinsic gradient updates

We compare convergence against the closed-form SVD solution.

