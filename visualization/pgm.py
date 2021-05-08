import daft
import matplotlib.pyplot as plt

plt.style.use(
    [
        "./visualization/stylesheets/science.mplstyle",
        "./visualization/stylesheets/probnum_colors.mplstyle",
        "./visualization/stylesheets/9pt.mplstyle",
    ]
)


pgm = daft.PGM(dpi=300)
pgm.add_plate(
    [-0.5, -1.5, 5, 2],
    rect_params={"ec": "None"},
)
pgm.add_plate(
    [-0.5, -0.5, 5, 1],
    rect_params={"ec": "C1"},
)


pgm.add_node("y0", r"$Y(t_0)$", 0, 0)
pgm.add_node("y1", r"...", 1, 0)
pgm.add_node("y2", r"$Y(t_n)$", 2, 0)
pgm.add_node("y3", r"...", 3, 0)
pgm.add_node("y4", r"$Y(t_N)$", 4, 0)

pgm.add_node("ell0", r"$\ell_0$", 0, 1, alternate=True)
pgm.add_node("ell2", r"$\ell_n$", 2, 1, alternate=True)
pgm.add_node("ell4", r"$\ell_N$", 4, 1, alternate=True)

pgm.add_node("ellL", r"$\ell_L$", 0, -1, alternate=True)
pgm.add_node("ellR", r"$\ell_R$", 4, -1, alternate=True)

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
ax.set_title(
    r"$\bf A$" + "  ", loc="left", fontweight="bold", ha="right", fontsize="x-large"
)

plt.savefig("./figures/pgm_attempt.pdf")
plt.show()