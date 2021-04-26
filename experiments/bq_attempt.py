"""How hard is it to make BQ work?"""

import numpy as np
from probnum import quad, kernels


def fun(x):
    """f(x) = sin(x) -- this function will be approximated with BQ."""
    return np.sin(x)


# Construct objects
gaussian_kernel = kernels.ExpQuad(input_dim=1, lengthscale=1.0)
lebesgue_measure = quad.LebesgueMeasure(domain=[0.0, 1.0])
kernel_embedding = quad.KernelEmbedding(gaussian_kernel, lebesgue_measure)


# Choose grid and compute kernel embeddings
grid = np.array([0.0, 0.33, 0.5, 0.67, 1.0]).reshape((-1, 1))
mu = kernel_embedding.kernel_mean(grid)
Sigma = kernel_embedding.kernel_variance()

# Compute weights and (uncalibrated) posterior variance
K = gaussian_kernel(grid, grid)
Kinv = np.linalg.inv(K)
weights = mu @ Kinv
variance = Sigma - mu @ Kinv @ mu

# Approximate a function: sin(x)
fx = fun(grid).squeeze()
sigma_squared = fx @ Kinv @ fx / grid.size
approx = weights @ fx
calibrated_variance = sigma_squared * variance


# Evaluate error and compare with calibrated standard deviation.
truth = -(np.cos(1) - np.cos(0))
abs_error = approx - truth
rel_error = np.abs(abs_error / truth)


# Print result
print("Relative error:", rel_error)
print()
print("Posterior standard deviation:", np.sqrt(calibrated_variance))
