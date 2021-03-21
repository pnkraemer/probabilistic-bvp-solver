"""Meshing."""


import numpy as np


def split_grid(a):
    """

    Examples
    --------
    >>> import numpy as np
    >>> a = np.arange(1, 5)
    >>> print(a)
    [1 2 3 4]
    >>> b = split_grid(a)
    >>> print(np.round(b, 1))
    [1.  1.3 1.7 2.  2.3 2.7 3.  3.3 3.7 4. ]
    """

    diff = np.diff(a)
    x = a[:-1] + diff * 1.0 / 3.0
    y = a[:-1] + diff * 2.0 / 3.0
    full = np.append(np.append(a, x), y)
    return np.sort(full)
