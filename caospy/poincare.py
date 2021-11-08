# import sympy as sp
# import pandas as pd

# ==============================================================================
# Docs
# ==============================================================================

"""Poincare method for periodic orbits implemented."""

import numpy as np

from . import trajectories


class Poincare(trajectories.Trajectory):
    """
    Implementation for Poincaré maps.

    Is initialized from the poincare method in the Functional
    class of the core module.


    Example
    -------
    >>> v = ['x', 'y', 'z']
    >>> f = ['sigma*(-x + y)', 'x*(rho - z) - y', '-beta*z + x*y']
    >>> p = ['sigma', 'rho', 'beta']
    >>> p_values = [166.04, 8/3, 10.0]
    >>> sample_sys = caospy.Symbolic(v, f, p, 'sample')
    >>> x0 = [2.0, -1.0, 150.0]
    >>> sample_poincare = sample_sys.poincare(x0, p_values)
    >>> sample_poincare
    <caospy.poincare.Poincare object at 0x000002112AB322E0>
    >>> sample_poincare.t
    array([0.00000000e+00, 4.00032003e-03, 8.00064005e-03, ...,
           4.99919994e+01, 4.99959997e+01, 5.00000000e+01])
    >>> sample_poincare.x
    array([[4.0824829 , 4.0824829 , 4.0824829 , ..., 4.0824829 , 4.0824829 ,
            4.0824829 ],
           [4.0824829 , 4.0824829 , 4.0824829 , ..., 4.0824829 , 4.0824829 ,
            4.0824829 ],
           [1.66666667, 1.66666667, 1.66666667, ..., 1.66666667, 1.66666667,
            1.66666667]])

    """

    def __init__(self, t, x, variables):
        super().__init__(t, x, variables)

    def _fit(self, a, plane, grade, axis):
        """
        Compute the polynomial fitting of the portion the trajectory.

        It crosses the desired Poincaré surface plane,
        over which the map will be evaluated.

        Parameters
        ----------
        a: ``int``
            Value of the plane to evaluate the trajectory and map it.
        plane: ``int``
            Name of the axis uppon which to set the plane.
        grade: ``int``
            Grade of the fitting polynomial.
        axis: ``int``
            Axis uppon which to perform the mapping.

        Return
        ------
        out: tuple
            Tuple of three objects. First it returns the time points where
            the trajectory touched the plane, second, it gives the xn+1,
            xn array, and finally the iterations array,
            which contains the nth iterations with their respective values.

        Example
        -------
        >>> v = ['x', 'y', 'z']
        >>> f = ['sigma*(-x + y)', 'x*(rho - z) - y', '-beta*z + x*y']
        >>> p = ['sigma', 'rho', 'beta']
        >>> sample_sys = caospy.Symbolic(v, f, p, 'sample')
        >>> p_values = [166.04, 8/3, 10.0]
        >>> x0 = [2.0, -1.0, 150.0]
        >>> sample_poincare = sample_sys.poincare(x0, p_values)

        """
        tc = []
        x2 = []
        points = len(self.t)
        x_plane = self.x[plane - 1]
        x_axis = self.x[axis - 1]
        for i in range(1, points):
            if x_plane[i - 1] < a < x_plane[i]:
                pa = np.polyfit(
                    x_plane[(i - 1) : (i + 2)],  # noqa
                    x_axis[(i - 1) : (i + 2)],  # noqa 
                    grade,
                )
                tc.append(self.t[i])
                x2.append(pa[0] * a ** 2 + pa[1] * a + pa[2])

        i_times_len = len(tc)
        mp = np.zeros((2, i_times_len))
        for i in range(1, i_times_len):
            mp[0, i] = x2[i - 1]
            mp[1, i] = x2[i]
        tc = np.array(tc)
        x2 = np.array(x2)

        return tc, mp, x2
# Noqa is for ignore the two lines because flake8 doesn't allow space
# before : operator, but it's necessary for black.

# class Map:
#    """Defines maps objects with iterations and values."""
#
#    def __init__(self, t, n, i, plane, axis):
#        self.n0 = n[0]
#        self.n1 = n[1]
#        self.iterations = i
#        self.t_iter = t
#        self.plane = plane
#        self.axis = axis
