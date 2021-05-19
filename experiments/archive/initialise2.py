"""How good is the initialisation function."""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from probnum import diffeq, filtsmooth
from probnum import random_variables as randvars
from probnum import randvars, statespace
from probnum._randomvariablelist import _RandomVariableList
from scipy.integrate import solve_bvp, solve_ivp
from tqdm import tqdm

from bvps import *

# Problem parameters

TMAX = 0.5
XI = 0.0001
bvp = problem_7_second_order(xi=XI)
bvp1st = problem_7(xi=XI)


# Algorithm parameters
TOL = 1e-1

q = 2

initial_grid = np.linspace(bvp.t0, bvp.tmax, 10)
initial_guess = np.zeros((2, len(initial_grid)))


# Reference solution
refsol = solve_bvp(bvp1st.f, bvp1st.scipy_bc, initial_grid, initial_guess, tol=TOL)


# Initialisation
ibm = statespace.IBM(
    ordint=q,
    spatialdim=2,
    forward_implementation="sqrt",
    backward_implementation="sqrt",
)
integ = WrappedIntegrator(ibm, bvp)


initial_grid = np.linspace(bvp.t0, bvp.tmax, 50)

posterior = bvp_initialise(
    bvp=bvp1st,
    bridge_prior=integ,
    initial_grid=initial_grid,
)

initial_guess = np.zeros((len(initial_grid), 2))
initial_guess[:, 1] = 2 * np.sin(1 * np.pi * (initial_grid + 1)) + ((initial_grid + 1))

posterior_guesses = bvp_initialise_guesses(
    bvp=bvp1st,
    bridge_prior=integ,
    initial_grid=initial_grid,
    initial_guesses=initial_guess,
)

posterior_ode = bvp_initialise_ode(
    bvp=bvp1st,
    bridge_prior=integ,
    initial_grid=initial_grid,
)


# Check initial mean
print(
    refsol.niter,
    len(refsol.x),
)
print(
    posterior.states[0].mean[:2],
    # bvp.solution(bvp.t0),
    refsol.y[:, 0],
)
print(
    posterior.states[-1].mean[:2],
    # bvp.solution(bvp.tmax),
    refsol.y[:, -1],
)

print("Done.")

initial_mean = posterior_ode.states[0].mean[:2]
sol = solve_ivp(
    bvp1st.f,
    t_span=(bvp1st.t0, bvp1st.t0 + 0.1),
    y0=initial_mean,
    atol=1e-6,
    rtol=1e-6,
    dense_output=True,
)
print(sol)

# Visualisation
evalgrid = np.linspace(bvp.t0, bvp.tmax, 250)
plt.style.use("fivethirtyeight")
plt.subplots(dpi=200)


plt.plot(evalgrid, sol.sol(evalgrid).T)
plt.plot(
    evalgrid,
    refsol.sol(evalgrid)[1],
    linestyle="dashed",
    alpha=0.5,
    label="Truth",
)
plt.plot(
    evalgrid,
    posterior_ode(evalgrid).mean[:, 3],
    alpha=0.5,
    label="Smoother",
)
plt.plot(
    evalgrid,
    posterior_ode.filtering_posterior(evalgrid).mean[:, 3],
    alpha=0.5,
    label="Filter",
)
for t in posterior.locations:
    plt.axvline(t, linewidth=1, color="gray")
plt.ylim((-3, 6))
plt.legend()
plt.show()


# np.save(
#     "./probabilistic-bvp-solver/data/initialisation_visualisation/evalgrid", evalgrid
# )


# np.save(
#     "./probabilistic-bvp-solver/data/initialisation_visualisation/initial_grid",
#     initial_grid,
# )


# np.save(
#     "./probabilistic-bvp-solver/data/initialisation_visualisation/initial_guess",
#     initial_guess,
# )


# np.save(
#     "./probabilistic-bvp-solver/data/initialisation_visualisation/truth",
#     refsol.sol(evalgrid)[1],
# )


# np.save(
#     "./probabilistic-bvp-solver/data/initialisation_visualisation/smoother_ode",
#     posterior_ode(evalgrid).mean[:, 3],
# )


# np.save(
#     "./probabilistic-bvp-solver/data/initialisation_visualisation/filter_ode",
#     posterior_ode.filtering_posterior(evalgrid).mean[:, 3],
# )


# np.save(
#     "./probabilistic-bvp-solver/data/initialisation_visualisation/smoother_guesses",
#     posterior_guesses(evalgrid).mean[:, 3],
# )


# np.save(
#     "./probabilistic-bvp-solver/data/initialisation_visualisation/filter_guesses",
#     posterior_guesses.filtering_posterior(evalgrid).mean[:, 3],
# )
