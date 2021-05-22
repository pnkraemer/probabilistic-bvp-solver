import time

import numpy as np
import tqdm
from probnum import diffeq, statespace
from probnumeval import timeseries
from scipy.integrate import solve_bvp

from bvps import bvp_solver, problem_examples
import matplotlib.pyplot as plt


# Load problems


bvp = problem_examples.problem_24_second_order()
bvp1st = bvp.to_first_order()

q = 6
tol = 1e-1

# Compute reference solution
initial_grid = np.linspace(bvp.t0, bvp.tmax, 50)
initial_guess = np.ones((bvp1st.dimension, len(initial_grid)))
refsol_fine = solve_bvp(
    bvp1st.f, bvp1st.scipy_bc, initial_grid, initial_guess, tol=1e-6
)
assert refsol_fine.success

initial_grid = np.linspace(bvp.t0, bvp.tmax, 5)
initial_guess = np.ones((bvp1st.dimension, len(initial_grid)))

bvp.solution = refsol_fine.sol
reference_solution = lambda *args: bvp.solution(*args)[0][:, None]
testlocations = np.linspace(bvp.t0, bvp.tmax, 400)


# Solve the BVP
ibm = statespace.IBM(
    ordint=q,
    spatialdim=bvp.dimension,
    forward_implementation="sqrt",
    backward_implementation="sqrt",
)
solver = bvp_solver.BVPSolver.from_default_values_std_refinement(
    ibm, initial_sigma_squared=1e8, normalise_with_interval_size=False
)

start_time = time.time()

initial_posterior, _ = solver.compute_initialisation(
    bvp, initial_grid, initial_guess=None, use_bridge=True
)

solution_gen = solver.solution_generator(
    bvp,
    atol=tol,
    rtol=tol,
    initial_posterior=initial_posterior,
    maxit_ieks=3,
    maxit_em=1,
    yield_ieks_iterations=False,
)

for (kalman_posterior, sigma_squared), _ in zip(solution_gen, range(8)):
    print(len(kalman_posterior.locations))
runtime = time.time() - start_time

ode_posterior = diffeq.KalmanODESolution(kalman_posterior)

print(ode_posterior.states.std)

print(len(ode_posterior.locations), len(refsol_fine.x))
plt.plot(refsol_fine.x, refsol_fine.y.T[:, 0])
plt.plot(ode_posterior.locations, ode_posterior.states.mean)
# plt.plot(ode_posterior.locations, ode_posterior.derivatives.mean)
plt.show()
