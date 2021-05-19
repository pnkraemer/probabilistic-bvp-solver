import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_bvp


def fun(x, y):

    return np.vstack((y[1], -np.exp(y[0])))


def bc(ya, yb):

    return np.array([ya[0], yb[0]])


x = np.linspace(0, 1, 5)


y_a = np.zeros((2, x.size))


refsol = solve_bvp(fun, bc, x, y_a, tol=1e-15)

grid = np.linspace(0, 1, 150)
errors = []
meshsizes = []
niters = []
for tol in 10.0 ** (-np.arange(12)):
    sol = solve_bvp(fun, bc, x, y_a, tol=tol)
    err = np.linalg.norm(sol.sol(grid) - refsol.sol(grid)) / np.sqrt(
        refsol.sol(grid).size
    )

    print(sol.x.shape)
    meshsizes.append(len(sol.x))
    niters.append(sol.niter)
    errors.append(err)
    print(err)


plt.plot(sol.x, sol.y[0])
for t in sol.x[::4]:
    plt.axvline(t, alpha=1)
plt.show()
plt.subplots(dpi=500)
for (e, x, i) in zip(errors, meshsizes, niters):
    plt.loglog(e, x, marker="o", markersize=2 * i, color="gray")
plt.xlabel("Errors")
plt.ylabel("Final mesh size")
plt.show()
print(errors)
