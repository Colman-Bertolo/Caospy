# import sympy as sp

# Systems trajectories treatment

# ==============================================================================
# Docs
# ==============================================================================

"""Treatment of integrated trajectories of dynamical systems."""

# ==============================================================================
# Imports
# ==============================================================================

import numpy as np

import pandas as pd

# ==============================================================================
# Class Trajectory
# ==============================================================================


class Trajectory:
    """Time series and states matrix given from an integrated dynamical system.

    The attributes defined for the object are the time vector,

    the state matrix, the number of variables and the column names.

    The time vector is a numpy.linspace defined by the

    integration time interval and the number of points or step defined.

    The state matrix has the system's

    variable values for every given time in the time vector.

    Attributes
    ----------
    x: ``np.array``
        Numpy array with shape (i, j). Being i the number of variables and j

        the number of points of integration.

    t: ``np.array``
        Numpy array with the time interval defined in the integration.
    n_var: ``int``
        Number of variables of the system.
    cols: ``list``
        List with column names for data frame of trajectories.

    Example
    -------
    >>> v = ['x', 'y']
    >>> f = ['a * y', 'b * x - c * y']
    >>> p = ['a', 'b', 'c']
    >>> sample_sys = caospy.Symbolic(v, f, p, 'sample')
    >>> p_values = [1, -2, 3]
    >>> x0 = [1, 1]
    >>> sample_trajectory = sample_sys.time_evolution(x0, p_values)
    >>> sample_trajectory.t
    array([0.00000e+00, 4.00008e-03, 8.00016e-03, ..., 1.99992e+02,
           1.99996e+02, 2.00000e+02])
    >>> sample_trajectory.x
    array([[ 1.00000000e+00,  1.00396022e+00,  1.00784126e+00, ...,
             4.18503698e-87,  4.16832993e-87,  4.15168958e-87],
           [ 1.00000000e+00,  9.80103295e-01,  9.60412752e-01, ...,
            -4.18503698e-87, -4.16832993e-87, -4.15168958e-87]])
    >>> sample_trajectory.n_var
    3
    """

    def __init__(self, t, x, variables):
        self.x = x  # Devuelve matriz de trayectorias x
        self.t = t  # Devuelve vector de tiempo t
        self.n_var = np.size(x[:, 0])
        if variables == []:
            variables = [f"$x_{i}$" for i in range(self.n_var)]
        self.cols = ["t"] + [f"{v}" for v in variables]

    def to_table(self):
        """Return a data frame of the trajectory, indicating variables states

           and time.

        Return
        ------
        out: pandas.DataFrame
            Pandas data frame with a row for every time point of the interval

            and a column for time and every variable of the system.

        Example
        -------
        >>> v = ['x1', 'x2', 'x3']
        >>> f = ['x1 * x2 - 2', '-x3 + 1 * (x2**2)', '3 * x1 - x2']
        >>> p = ['a', 'b', 'c']
        >>> p = []
        >>> p_values = []
        >>> sample_sys = caospy.Symbolic(v, f, p, 'sample')
        >>> x0 = [1, 1, 1]
        >>> sample_trajectory = sample_sys.time_evolution(x0, p_values)
        >>> sample_trajectory.to_table()
                      t        x1        x2        x3
        0      0.00e+00  1.00e+00  1.00e+00  1.00e+00
        1      4.00e-03  9.96e-01  1.00e+00  1.01e+00
        2      8.00e-03  9.92e-01  1.00e+00  1.02e+00
        3      1.20e-02  9.88e-01  1.00e+00  1.02e+00
        4      1.60e-02  9.84e-01  1.00e+00  1.03e+00
        ...         ...       ...       ...       ...
        49995  2.00e+02 -1.47e+73  1.19e+73 -2.33e+60
        49996  2.00e+02 -1.47e+73  1.19e+73 -2.33e+60
        49997  2.00e+02 -1.47e+73  1.19e+73 -2.33e+60
        49998  2.00e+02 -1.47e+73  1.19e+73 -2.33e+60
        49999  2.00e+02 -1.47e+73  1.19e+73 -2.33e+60
        """
        col_names = self.cols
        pd.set_option("display.precision", 2)
        merge = np.vstack((self.t, self.x))
        trajectory_table = pd.DataFrame(merge.T, columns=col_names)
        return trajectory_table
