import daft
import matplotlib.pyplot as plt


plt.rcParams["text.usetex"] = True

plt.style.use(
    [
        "./visualization/stylesheets/fontsize/7pt.mplstyle",
        "./visualization/stylesheets/color/probnum_colors.mplstyle",
    ]
)
pgm = daft.PGM(dpi=300, aspect=1.0, grid_unit=1.41)

# Bridge prior
pgm.add_plate(
    [-0.5, -1.5, 5.0, 2.0],
    rect_params={"ec": "C1", "linewidth": 2},
)

# Classic prior
pgm.add_plate(
    [-0.55, -0.55, 5.10, 1.10],
    rect_params={"ec": "C0", "linewidth": 2},
)

# Placeholder for space reasons
pgm.add_plate(
    [-0.55, -1.85, 5.0, 2.0],
    rect_params={"ec": "None", "linewidth": 2},
)


pgm.add_node("y0", r"$Y(t_0)$", 0, 0)
pgm.add_node("y1", r"...", 1.0, 0, plot_params={"ec": "None"})
pgm.add_node("y2", r"$Y(t_n)$", 2, 0)
pgm.add_node("y3", r"...", 3, 0, plot_params={"ec": "None"})
pgm.add_node("y4", r"$Y(t_N)$", 4, 0)

pgm.add_node("ell0", r"$\ell_0$", 0, 1.0, alternate=True)
pgm.add_node("ell2", r"$\ell_n$", 2, 1.0, alternate=True)
pgm.add_node("ell4", r"$\ell_N$", 4, 1.0, alternate=True)

pgm.add_node("ellL", r"$\ell_L$", 0, -1.0, alternate=True)
pgm.add_node("ellR", r"$\ell_R$", 4, -1.0, alternate=True)

pgm.add_edge("y0", "y1")
pgm.add_edge("y1", "y2")
pgm.add_edge("y2", "y3")
pgm.add_edge("y3", "y4")

pgm.add_edge("y0", "ell0")
pgm.add_edge("y2", "ell2")
pgm.add_edge("y4", "ell4")

pgm.add_edge("y0", "ellL")
pgm.add_edge("y4", "ellR")


ax = pgm.render()
# ax.set_title(
#     r"$\bf A$" + "  ",
#     loc="left",
#     fontweight="bold",
#     ha="right",
#     fontsize="x-large",
#     pad=-20,
# )

plt.savefig("./figures/pgm.pdf")
plt.show()
