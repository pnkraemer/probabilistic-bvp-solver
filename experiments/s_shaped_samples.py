import numpy as np
import matplotlib.pyplot as plt

# # This function evaluates the differential equation c'' = f(c, c')
# def geodesic_system(manifold, c, dc):
#     # Input: c, dc ( D x N )

#     D, N = c.shape
#     if (dc.shape[0] != D) | (dc.shape[1] != N):
#         print('geodesic_system: second and third input arguments must have same dimensionality\n')
#         sys.exit(1)

#     # Evaluate the metric and the derivative
#     M, dM = manifold.metric_tensor(c, nargout=2)

#     # Prepare the output (D x N)
#     ddc = np.zeros((D, N))

#     # Diagonal Metric Case, M (N x D), dMdc_d (N x D x d=1,...,D) d-th column derivative with respect to c_d
#     if manifold.is_diagonal():
#         for n in range(N):
#             dMn = np.squeeze(dM[n, :, :])
#             ddc[:, n] = -0.5 * (2 * np.matmul(dMn * dc[:, n].reshape(-1, 1), dc[:, n])
#                                 - np.matmul(dMn.T, (dc[:, n] ** 2))) / M[n, :]

#     # Non-Diagonal Metric Case, M ( N x D x D ), dMdc_d (N x D x D x d=1,...,D)
#     else:

#         for n in range(N):
#             Mn = np.squeeze(M[n, :, :])
#             if np.linalg.cond(Mn) < 1e-15:
#                 print('Ill-condition metric!\n')
#                 sys.exit(1)

#             dvecMdcn = dM[n, :, :, :].reshape(D * D, D, order='F')
#             blck = np.kron(np.eye(D), dc[:, n])

#             ddc[:, n] = -0.5 * (np.linalg.inv(Mn) @ (
#                     2 * blck @ dvecMdcn @ dc[:, n]
#                     - dvecMdcn.T @ np.kron(dc[:, n], dc[:, n])))

#     return ddc
import scipy.linalg

N = 1000
a = 2 * np.random.rand(N, 2) - 1
x = np.abs(np.random.rand(N) * 0.2 - 0.0)

b = a / (np.linalg.norm(a, axis=1) + x).reshape((-1, 1))


plt.style.use(
    [
        "./visualization/stylesheets/science.mplstyle",
        "./visualization/stylesheets/misc/grid.mplstyle",
        # "./visualization/stylesheets/color/high-contrast.mplstyle",
        "./visualization/stylesheets/9pt.mplstyle",
        "./visualization/stylesheets/one_of_13_tile.mplstyle",
        "./visualization/stylesheets/hollow_markers.mplstyle",
        "./visualization/stylesheets/probnum_colors.mplstyle",
    ]
)
fig, ax = plt.subplots(ncols=1, constrained_layout=True)

plt.scatter(b[:, 0], b[:, 1], color="gray", alpha=0.5, s=2)
ax.set_title(r"$\bf C$" + "  ", loc="left", fontweight="bold", ha="right")

# plt.xlim((-5, 60))
# plt.ylim((0, 110))
plt.xlabel(r"Coordinate $x_1$")
plt.ylabel(r"Coordinate $x_2$")


plt.savefig("./figures/geodesics_plots.pdf")
plt.show()
