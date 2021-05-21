"""How hard is it to make BQ work?"""

import numpy as np
from probnum import kernels, quad

SCALE = 1e-3


def fun(x):
    """f(x) = sin(x) -- this function will be approximated with BQ."""
    return SCALE * np.sin(x)


# Construct objects
gaussian_kernel = kernels.ExpQuad(input_dim=1, lengthscale=1.0)
lebesgue_measure = quad.LebesgueMeasure(domain=[0.0, 1.0])
kernel_embedding = quad.KernelEmbedding(gaussian_kernel, lebesgue_measure)


# Choose grid and compute kernel embeddings
grid = np.array([0.0, 2.0 / 6.0, 3.0 / 6.0, 4.0 / 6.0, 1.0]).reshape((-1, 1))
mean_embedding = kernel_embedding.kernel_mean(grid)
variance_embedding = kernel_embedding.kernel_variance()

# Compute weights and (uncalibrated) posterior variance
K = gaussian_kernel(grid, grid)
Kinv = np.linalg.inv(K)
weights = mean_embedding @ Kinv
variance = np.abs(variance_embedding - weights @ mean_embedding)


# Approximate a function: sin(x)
fx = fun(grid).squeeze()
sigma_squared = fx @ Kinv @ fx / grid.size
approx = weights @ fx
calibrated_variance = sigma_squared * variance

# Evaluate error and compare with calibrated standard deviation.
truth = -SCALE * (np.cos(1) - np.cos(0))
abs_error = approx - truth
rel_error = np.abs(abs_error / truth)


# Print result
print("Relative error:", rel_error)
print("Absolute error:", abs_error)
print("Truth:", truth)
print("Approximation:", approx)
print()
print("Calibrated posterior standard deviation:", np.sqrt(calibrated_variance))
print("Uncalibrated posterior standard deviation:", np.sqrt(variance))
print("Calibrated diffusion:", np.sqrt(sigma_squared))
