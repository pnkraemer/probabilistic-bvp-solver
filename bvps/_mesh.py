"""Meshing."""


import numpy as np


# def split_grid(a):
#     """

#     Examples
#     --------
#     >>> import numpy as np
#     >>> a = np.arange(1, 5)
#     >>> print(a)
#     [1 2 3 4]
#     >>> b = split_grid(a)
#     >>> print(np.round(b, 1))
#     [1.  1.3 1.7 2.  2.3 2.7 3.  3.3 3.7 4. ]
#     """

#     diff = np.diff(a)
#     x = a[:-1] + diff * 1.0 / 3.0
#     y = a[:-1] + diff * 2.0 / 3.0
#     full = np.append(np.append(a, x), y)
#     return np.sort(full)


# def new_grid(a):
#     diff = np.diff(a)
#     x = a[:-1] + diff * 1.0 / 3.0
#     y = a[:-1] + diff * 2.0 / 3.0
#     full = np.append(x, y)
#     return np.sort(full)


# def insert_two_points(a):
#     diff = np.diff(a)
#     x = a[:-1] + diff * 1.0 / 3.0
#     y = a[:-1] + diff * 2.0 / 3.0
#     full = np.append(x, y)
#     return np.sort(full), np.repeat(diff, 2)


# def insert_three_points(a):
#     diff = np.diff(a)
#     x = a[:-1] + diff * 1.0 / 5.0
#     y = a[:-1] + diff * 2.0 / 5.0
#     z = a[:-1] + diff * 3.0 / 5.0
#     w = a[:-1] + diff * 4.0 / 5.0
#     full = np.append(x, y)
#     full2 = np.append(full, z)
#     full3 = np.append(full2, w)
#     return np.sort(full3), np.repeat(diff, 4)


# def new_grid2(a):
#     diff = np.diff(a)
#     x = a[:-1] + diff * 1.0 / 2.0
#     return np.sort(x)


# def insert_single_points(a):
#     diff = np.diff(a)
#     x = a[:-1] + diff * 1.0 / 2.0
#     return np.sort(x), diff


def insert_lobatto5_points(mesh, where):
    """Insert 5-pt Lobatto points into a mesh."""
    diff = np.diff(mesh)

    left = (np.sqrt(7.0) - np.sqrt(3)) / np.sqrt(28)
    middle = 1.0 / 2.0
    right = (np.sqrt(7.0) + np.sqrt(3)) / np.sqrt(28)

    mesh_left = mesh[:-1] + diff * left
    mesh_middle = mesh[:-1] + diff * middle
    mesh_right = mesh[:-1] + diff * right

    mesh_left_and_middle = np.union1d(mesh_left[where], mesh_middle[where])
    lobatto = np.union1d(mesh_left_and_middle, mesh_right[where])
    new_mesh = np.union1d(mesh, lobatto)

    return new_mesh, lobatto, np.repeat(diff, 3)


def insert_central_point(mesh, where):
    diff = np.diff(mesh)
    central = mesh[:-1] + diff * 1.0 / 2.0
    new_mesh = np.union1d(mesh, central[where])

    return new_mesh, central, diff


def insert_two_equispaced_points(mesh, where):
    diff = np.diff(mesh)
    x = mesh[:-1] + diff * 1.0 / 3.0
    y = mesh[:-1] + diff * 2.0 / 3.0
    new_points = np.union1d(x[where], y[where])
    new_mesh = np.union1d(mesh, new_points)
    return new_mesh, new_points, np.repeat(diff, 2)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    arr = np.linspace(0, 1, 2)
    arr_with_central, *_ = insert_central_point(arr, where=[True])
    arr_with_two, *_ = insert_two_equispaced_points(arr)
    arr_with_lobatto, *_ = insert_lobatto5_points(arr)

    arr_with_central2, *_ = insert_central_point(
        arr_with_lobatto, where=(arr_with_lobatto < 0.5).nonzero()
    )

    y1 = np.ones_like(arr)
    y2 = 1.1 * np.ones_like(arr_with_central)
    y3 = 1.2 * np.ones_like(arr_with_two)
    y4 = 1.3 * np.ones_like(arr_with_lobatto)
    y5 = 1.4 * np.ones_like(arr_with_central2)

    plt.subplots(dpi=200)
    plt.plot(arr, y1, "o", color="black")
    plt.plot(arr_with_central, y2, "x", color="black")
    plt.plot(arr_with_two, y3, "+", color="black")
    plt.plot(arr_with_lobatto, y4, "d", color="black")
    plt.plot(arr_with_central2, y5, "d", color="red")
    plt.ylim((0, 2))
    plt.show()