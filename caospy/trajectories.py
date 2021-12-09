# import sympy as sp

# Systems trajectories treatment

# ==============================================================================
# Docs
# ==============================================================================

"""Treatment of integrated trajectories of dynamical systems."""

# ==============================================================================
# Imports
# ==============================================================================

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

# ==============================================================================
# Class Trajectory
# ==============================================================================


class Trajectory:
    """
    Time series and states matrix given from an integrated dynamical system.

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

    """

    def __init__(self, t, x, variables):
        self.x = x  # Devuelve matriz de trayectorias x
        self.t = t  # Devuelve vector de tiempo t
        self.n_var = np.size(x[:, 0])
        if variables == []:
            variables = [f"$x_{i}$" for i in range(self.n_var)]
        self.cols = ["t"] + [f"{v}" for v in variables]
        self.variables = variables

    def to_table(self):
        """

        Return a data frame of the trajectory.

        Indicates variables states and time.

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

    def plot_trajectory(self, var="t-x", ax=None, kws=None):
        """
        Plot the trajectories in 2D or 3D.

        It uses the variables from the Trajectories Class init and
        plot the numerical variables obtained.
        The user can to set up all the sizes, colors, vars to plot,
        etc.

        Parameters
        ----------
        var:``string``
             Variables to plot, you can use two (2D plot) o three (3D plot).
             They must be delimiter by `-` symbol.
        ax: ``matplotlib.pyplot.Axis``, (optional)
            Matplotlib axis specification.
        kws:```dict``  (optional)
            The parameters to send to set up the plot, like line colors,
            linewidth, marker face color, etc.
            Here is a list of available line properties:
            matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html


        Returns
        -------
        ``Axes3DSubplot`` or ``AxesSubplot``
            A configuration representing the axis data.

        Example
        -------
        >>> plot = plot_trajectory("x-y-z", plt.axes(projection ='3d'))
        Axes3DSubplot
        """
        # Check array's dimension
        if self.x.ndim == 1:
            name_var = dict([("t", self.t), ("x", self.x)])
        elif self.x.ndim == 2 and self.x.shape[1] == 2:
            name_var = dict(
                [("t", self.t), ("x", self.x[0, :]), ("y", self.x[1, :])]
            )
        else:
            name_var = dict(
                [
                    ("t", self.t),
                    ("x", self.x[0, :]),
                    ("y", self.x[1, :]),
                    ("z", self.x[2, :]),
                ]
            )

        ax = plt.gca() if ax is None else ax

        kws = {} if kws is None else kws

        kws.setdefault("color", "r")  # Red line
        kws.setdefault("lw", "2")  # Linewidth 2 points
        vars_plot = var.split("-")  # Split string vars to plot

        if len(vars_plot) == 3:  # Plot 3D
            ax.plot3D(
                name_var[vars_plot[0]],
                name_var[vars_plot[1]],
                name_var[vars_plot[2]],
                **kws,
            )
            ax.grid()
        else:
            ax.plot(name_var[vars_plot[0]], name_var[vars_plot[1]], **kws)
            ax.grid()

        return ax
