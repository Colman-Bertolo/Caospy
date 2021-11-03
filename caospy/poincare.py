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
        if not plane >= 1 and plane < 4:
            raise ValueError("Specified plane must be between 1 and 3")
        if not axis >= 1 and axis < 4:
            raise ValueError("Specified axis must be between 1 and 3")

        t = np.delete(self.t, [-1, -2])

        x1a = np.delete(np.roll(self.x[:, axis - 1], 2), [0, 1])
        x1b = np.delete(np.roll(self.x[:, axis - 1], 1), [0, 1])
        x1c = np.delete(self.x[:, axis - 1], [0, 1])

        x1_slices = np.vstack((x1a, x1b, x1c))

        x2a = np.delete(np.roll(self.x[:, plane - 1], 2), [0, 1])
        x2b = np.delete(np.roll(self.x[:, plane - 1], 1), [0, 1])
        x2c = np.delete(self.x[:, plane - 1], [0, 1])

        x2_slices = np.vstack((x2a, x2b, x2c))

        x1 = x1_slices[:, (x1_slices[0] < a) & (x1_slices[1] > a)]
        x2 = x2_slices[:, (x1_slices[0] < a) & (x1_slices[1] > a)]
        t_map = t[(x1_slices[0] < a) & (x1_slices[1] > a)]

        x12 = np.vstack((x2, x1))

        def poly(v, grade_poly):
            return np.polyfit(v[3:6], v[0:3], grade_poly)

        x12_coeff = np.apply_along_axis(poly, 0, x12, grade)

        def apply_poly(p, a_value):
            return (
                p[0] * a_value ** 2 + p[1] * a_value + p[2]
                if len(p) == 3
                else p[0] * a_value + p[1]
            )

        x2_fit = np.apply_along_axis(apply_poly, 0, x12_coeff, a)

        x21 = np.delete(x2_fit, -1)
        x22 = np.delete(x2_fit, 0)

        xmap = np.array([[x21], [x22]])

        return t_map, xmap, x2


class Map:
    """Defines maps objects with iterations and values."""

    def __init__(self, t, n, i, plane, axis):
        self.n0 = n[0]
        self.n1 = n[1]
        self.iterations = i
        self.t_iter = t
        self.plane = plane
        self.axis = axis
