import numpy as np
import matplotlib.pyplot as plt
from grassmann import grassmann_step
from objectives import pca_objective


def generate_data(n=50, samples=1000):
    """
    Generate synthetic data with structured covariance.
    """
    true_basis = np.random.randn(n, 3)
    true_basis, _ = np.linalg.qr(true_basis)

    coeffs = np.random.randn(3, samples)
    noise = 0.1 * np.random.randn(n, samples)

    X = true_basis @ coeffs + noise
    return X


def run_experiment():
    n = 50
    k = 3
    X = generate_data(n=n)

    # Covariance matrix
    S = (X @ X.T) / X.shape[1]

    loss_fn, grad_fn = pca_objective(S)

    # Random initialization
    Q = np.random.randn(n, k)
    Q, _ = np.linalg.qr(Q)

    step_size = 0.1
    iterations = 200
    losses = []

    for _ in range(iterations):
        Q = grassmann_step(Q, grad_fn, step_size)
        losses.append(loss_fn(Q))

    # Compare with SVD solution
    U, _, _ = np.linalg.svd(S)
    Q_svd = U[:, :k]
    optimal_value = loss_fn(Q_svd)

    print("Final Grassmann loss:", losses[-1])
    print("Optimal SVD loss:", optimal_value)

    plt.plot(losses)
    plt.title("Grassmann Optimization Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()


if __name__ == "__main__":
    run_experiment()
