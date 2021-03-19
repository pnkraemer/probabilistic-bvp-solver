import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from probnum import statespace, randvars, filtsmooth
from bvps import bratus, BoundaryValueProblem, WrappedIntegrator
from tqdm import tqdm


def from_ode(ode, prior):

    spatialdim = prior.spatialdim
    h0 = prior.proj2coord(coord=0)
    h1 = prior.proj2coord(coord=1)

    def dyna(t, x):
        return h1 @ x - ode.f(t, h0 @ x)

    def diff_cholesky(t):
        return 0.0 * np.eye(spatialdim)

    def jacobian(t, x):
        return h1 - ode.df(t, h0 @ x) @ h0

    discrete_model = statespace.DiscreteGaussian(
        input_dim=prior.dimension,
        output_dim = spatialdim,
        state_trans_fun=dyna,
        proc_noise_cov_mat_fun=diff_cholesky,
        jacob_state_trans_fun=jacobian,
        proc_noise_cov_cholesky_fun=diff_cholesky,
    )
    return filtsmooth.DiscreteEKFComponent(
        discrete_model,
        forward_implementation="sqrt",
        backward_implementation="sqrt",
    )


from scipy.integrate import solve_bvp



bvp = bratus(tmax=1.0)

initial_grid = np.linspace(bvp.t0, bvp.tmax, 500)
initial_guess = np.zeros((2, len(initial_grid)))
refsol = solve_bvp(bvp.f, bvp.scipy_bc, initial_grid, initial_guess, tol=1e-15)





q = 4
num_gridpoints = 50

ibm = statespace.IBM(
    ordint=q,
    spatialdim=2,
    forward_implementation="sqrt",
    backward_implementation="sqrt",
)

integ = WrappedIntegrator(ibm, bvp)


rv = randvars.Normal(np.zeros(ibm.dimension), 2.0 * np.eye(ibm.dimension))
initrv, _ = integ.forward_rv(rv, t=bvp.t0, dt=0.)

measmod = from_ode(bvp, ibm)

stopcrit = filtsmooth.StoppingCriterion(atol=1e-3, rtol=1e-3, maxit=500)

measmod_iterated = filtsmooth.IteratedDiscreteComponent(measmod, stopcrit=stopcrit)
kalman = filtsmooth.Kalman(dynamics_model=integ, measurement_model=measmod, initrv=initrv)
kalman_iterated = filtsmooth.Kalman(dynamics_model=integ, measurement_model=measmod_iterated, initrv=initrv)
P0 = ibm.proj2coord(0)
evalgrid = np.sort(np.random.rand(234))

for num_gridpoints in [1, 5, 10,50, 100, 500]:
    grid = np.linspace(bvp.t0, bvp.tmax, num_gridpoints)
    data = np.zeros((len(grid), 2))
    out = kalman.iterated_filtsmooth(dataset=data, times=grid, stopcrit=stopcrit)
    out_noniterated = kalman.filtsmooth(dataset=data, times=grid)
    out_3 = kalman_iterated.iterated_filtsmooth(dataset=data, times=grid, stopcrit=stopcrit)
    out_4 = kalman_iterated.filtsmooth(dataset=data, times=grid)

    x1 = (refsol.sol(evalgrid)).T
    x2 = (out(evalgrid).mean) @ P0.T
    x3 = (out_noniterated(evalgrid).mean) @ P0.T
    x4 = (out_3(evalgrid).mean) @ P0.T
    x5 = (out_4(evalgrid).mean) @ P0.T
    error = np.linalg.norm(x1  - x2) / np.sqrt(x2.size)
    error_noniterated = np.linalg.norm(x1  - x3) / np.sqrt(x3.size)
    error_3 = np.linalg.norm(x1  - x4) / np.sqrt(x3.size)
    error_4 = np.linalg.norm(x1  - x5) / np.sqrt(x3.size)
    print(f"N: {num_gridpoints} | IEKS-EKF: {error} | KS-EKS: {error_noniterated} | IEKS-IEKF: {error_3} | KS-IEKF: {error_4} ")


# plt.plot(out.locations, out.state_rvs.mean[:, 0])
# plt.show()