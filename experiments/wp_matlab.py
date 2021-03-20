import numpy as np


from scipy.integrate import solve_bvp

import matplotlib.pyplot as plt
from bvps import matlab_example

from probnumeval.timeseries import root_mean_square_error


bvp = matlab_example()

x = np.linspace(bvp.t0, bvp.tmax)


y_a = np.zeros((2, x.size))


refsol = solve_bvp(bvp.f, bvp.scipy_bc, x, y_a, tol=1e-15)

grid = np.linspace(0, 1, 150)
errors = []
meshsizes = []
niters = []
for tol in 10.0 ** (-np.arange(3, 12, 2)):
    sol = solve_bvp(bvp.f, bvp.scipy_bc, x, y_a, tol=tol)

    error = np.linalg.norm(sol.sol(grid) - refsol.sol(grid)) / np.sqrt(
        refsol.sol(grid).size
    )

    meshsizes.append(len(sol.x))
    niters.append(sol.niter)
    errors.append(error)

    plt.plot(sol.x, sol.y[0])
    for t in sol.x[::4]:
        plt.axvline(t, alpha=1)
    plt.show()

plt.subplots(dpi=100)
plt.loglog(meshsizes, errors)
plt.xlabel("Errors")
plt.ylabel("Final mesh size")
plt.show()


print()
print(errors)
