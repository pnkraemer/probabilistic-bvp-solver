from bvps import problem_examples, bvp_solver, ode_measmods
from scipy.integrate import solve_bvp
import numpy as np
import matplotlib.pyplot as plt

from probnum import diffeq, statespace

bvp = problem_examples.measles()


initial_grid = np.linspace(bvp.t0, bvp.tmax, 6)
initial_guess = np.zeros((6, len(initial_grid)))
initial_guess[0] = 0.07
initial_guess[1] = 0.001
initial_guess[2] = 0.001
initial_guess[3] = 0.07
initial_guess[4] = 0.001
initial_guess[5] = 0.001
refsol = solve_bvp(bvp.f, bvp.scipy_bc, initial_grid, initial_guess, tol=1e-5)


ibm = statespace.IBM(
    ordint=3,
    spatialdim=bvp.dimension,
    forward_implementation="sqrt",
    backward_implementation="sqrt",
)
measmod = ode_measmods.from_ode(bvp, ibm)

MAXIT = 250
fig, axes = plt.subplots(
    figsize=(3, 2),
    nrows=2,
    ncols=2,
    dpi=300,
    constrained_layout=True,
    sharex="col",
    sharey="row",
)


t = np.linspace(bvp.t0, bvp.tmax, 200)
for MAXIT_IEKS, axis_col in zip([MAXIT, 5], axes.T):
    # Solver and solver parameters
    solver = bvp_solver.BVPSolver.from_default_values_std_refinement(
        ibm, use_bridge=True, initial_sigma_squared=1e5
    )
    TOL = 1e-3

    solution_gen = solver.solution_generator(
        bvp,
        atol=TOL,
        rtol=TOL,
        initial_grid=initial_grid,
        initial_guess=initial_guess.T,
        maxit_ieks=MAXIT_IEKS,
        maxit_em=MAXIT // MAXIT_IEKS,
        yield_ieks_iterations=False,
    )
    # Skip initialisation
    reference_posterior, _ = next(solution_gen)
    reference_posterior, _ = next(solution_gen)

    posterior = diffeq.KalmanODESolution(reference_posterior)

    x = t
    y = reference_posterior(t).mean

    res = np.array(
        [
            measmod.forward_realization(y_, t=x_)[0].mean
            for (x_, y_) in zip(
                x, y
            )
        ]
    )

    if refsol.success:
        pass
        # plt.plot(refsol.x, refsol.y.T[:, :3], color="gray")
    else:
        plt.title("SciPy failed.")

    print(np.mean(y[:, 0]))
    # for t in x:
    #     plt.axvline(t, linewidth=0.5, color="k")
    # plt.plot(x, y[:, 0])
    # plt.plot(x, y[:, 1:3])
    axis_col[0].plot(x, y[:, 0] - np.mean(y[:, 0]), color="black")
    axis_col[0].plot(x, y[:, 1:3], color="black")
    axis_col[1].semilogy(x, np.abs(res), ".", nonpositive="clip")
    axis_col[0].set_ylim((-0.02, 0.02))
    axis_col[1].set_ylim((1e-20, 1e-4))

axes[0][0].set_title("IEKS")
axes[0][1].set_title("EM")
plt.show()
