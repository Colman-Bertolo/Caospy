# ==============================================================================
# Docs
# ==============================================================================

"""Poincare method for periodic orbits implemented."""

# ==============================================================================
# Imports
# ==============================================================================

import matplotlib.pyplot as plt

import numpy as np

from . import trajectories

# ==============================================================================
# Class Poincare
# ==============================================================================


class Poincare(trajectories.Trajectory):
    """Implementation for Poincaré maps.

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

    def _fit(self, axis, fixed, a, grade):
        tc = []
        x2 = []
        points = len(self.t)
        x_axis = self.x[axis - 1]
        x_fixed = self.x[fixed - 1]
        for i in range(1, points):
            if x_fixed[i - 1] < a < x_fixed[i]:
                pa = np.polyfit(
                    x_fixed[i - 1 : i + 2], x_axis[i - 1 : i + 2], grade # noqa
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

    def map_v(self, var, fixed, a=0, grade=2):
        """
        Compute the Poincaré map of a trajectory.

        Parameters
        ----------
        var: ``str``, ``int``
            Variable name or number whose map we want to generate.
        fixed: ``str``, ``int``
            Variable name or number to fix a surface across which
            the stability will be evaluated.
        a: ``float``
            Value at which we want to fix the variable.
        grade: ``int`` in (1, 2)
            Grade to fit the polynomial across the fixed surface.

        Return
        ------
        map_var: ``Map``
            Instance of class map.

        Example
        -------
        >>> import caospy as cp
        >>> lorenz = cp.Lorenz()
        >>> x0 = [2.0, -1.0, 1.50]
        >>> p = [10.0, 166.04, 8/3]
        >>> poincare_lorenz = lorenz.poincare(x0, p, t_disc=500, t_calc=40)
        >>> z_map = poincare_lorenz.map_v("z", "x")
        >>> z_map.n0
        array([ 0.        , 41.28553749, 41.2855924 , 41.28465165, 41.28434672,
            41.28455364, 41.28513428, 41.28593716, 41.28505819, 41.28443452,
            41.28438951, 41.28479233, 41.28549839, 41.28565587, 41.28468154,
            41.28434863, 41.2845339 , 41.28509986, 41.28589569, 41.28510502,
            41.28445035, 41.28438028, 41.28476469, 41.28545963, 41.28572109,
            41.28471292, 41.28435175, 41.28451505, 41.28506599, 41.28585437,
            41.28515349, 41.28446754, 41.28437212, 41.28473779, 41.28542123])

        """
        if var == fixed:
            raise ValueError(
                "'var' and 'fixed' must represent "
                "different variables, but are equal."
            )

        if isinstance(var, str):
            if var in self.variables:
                plane = self.variables.index(var) + 1

            else:
                raise ValueError(
                    f"'{var}' is not a system's variable."
                    f"var should be in {self.variables}."
                )
        elif isinstance(var, int):
            if var in [i for i in range(len(self.variables))]:
                plane = var + 1

            else:
                v = [1, 2, 3]
                raise ValueError(
                    f"'var' is out of range, " f"should be in between ({v})"
                )

        if isinstance(fixed, str):
            if fixed in self.variables:
                fixed = self.variables.index(fixed) + 1

            else:
                raise ValueError(
                    f"'{fixed}' is not a system's variable."
                    f"fixed should be in {self.variables}."
                )
        elif isinstance(fixed, int):
            if fixed in [i for i in range(len(self.variables))]:
                fixed = fixed + 1

            else:
                v = [1, 2, 3]
                raise ValueError(
                    f"'fixed' is out of range, " f"should be in between ({v})"
                )

        axis = list(set([1, 2, 3]) - set([plane, fixed]))[0]
        fit = self._fit(axis, fixed, a, grade)
        var_names = [self.variables[plane - 1], self.variables[axis - 1]]
        map_var = Map(fit[0], fit[1], fit[2], var_names)
        return map_var


# ==============================================================================
# Class Map
# ==============================================================================


class Map:
    """Defines maps objects with iterations and values.

    Attributes
    ----------
    n0: ``np.array``
        n-th values of the variable map.
    n1: ``np.array``
        n+1-th values of the variable map.
    iterations: ``np.array``
        Values of the variable when intersecting the hyper-surface.
    t_iter: ``np.array``
        Values of the times when the trajectory intersects the
        hyper-surface.
    var_names: ``list``
        List of strings, whose first element is the name of the
        mapped variable and the second one is the name of the
        fitted variable.


    Example
    -------

    >>> import caospy as cp
    >>> lorenz = cp.Lorenz()
    >>> x0 = [2.0, -1.0, 1.50]
    >>> p = [10.0, 166.04, 8/3]
    >>> poincare_lorenz = lorenz.poincare(x0, p, t_disc=500, t_calc=40)
    >>> z_map = poincare_lorenz.map_v("z", "x")
    >>> z_map.n0
    array([ 0.        , 41.28553749, 41.2855924 , 41.28465165, 41.28434672,
           41.28455364, 41.28513428, 41.28593716, 41.28505819, 41.28443452,
           41.28438951, 41.28479233, 41.28549839, 41.28565587, 41.28468154,
           41.28434863, 41.2845339 , 41.28509986, 41.28589569, 41.28510502,
           41.28445035, 41.28438028, 41.28476469, 41.28545963, 41.28572109,
           41.28471292, 41.28435175, 41.28451505, 41.28506599, 41.28585437,
           41.28515349, 41.28446754, 41.28437212, 41.28473779, 41.28542123])
    >>> z_map.n1
    array([ 0.        , 41.2855924 , 41.28465165, 41.28434672, 41.28455364,
           41.28513428, 41.28593716, 41.28505819, 41.28443452, 41.28438951,
           41.28479233, 41.28549839, 41.28565587, 41.28468154, 41.28434863,
           41.2845339 , 41.28509986, 41.28589569, 41.28510502, 41.28445035,
           41.28438028, 41.28476469, 41.28545963, 41.28572109, 41.28471292,
           41.28435175, 41.28451505, 41.28506599, 41.28585437, 41.28515349,
           41.28446754, 41.28437212, 41.28473779, 41.28542123, 41.28578807])
    >>> z_map.t_iter
    array([ 0.96009601,  2.09620962,  3.22832283,  4.36043604,  5.49254925,
            6.62466247,  7.75677568,  8.89288929, 10.0250025 , 11.15711571,
           12.28922892, 13.42134213, 14.55745575, 15.68956896, 16.82168217,
           17.95379538, 19.08590859, 20.2180218 , 21.35413541, 22.48624862,
           23.61836184, 24.75047505, 25.88258826, 27.01870187, 28.15081508,
           29.28292829, 30.4150415 , 31.54715472, 32.67926793, 33.81538154,
           34.94749475, 36.07960796, 37.21172117, 38.34383438, 39.47994799])
    >>> z_map.var_names
    ["z", "y"]

    """

    def __init__(self, t, n, i, var_names):
        self.n0 = n[0]
        self.n1 = n[1]
        self.iterations = i
        self.t_iter = t
        self.var_names = var_names

    def plot_cobweb(self, ax=None, kws_scatter=None, kws_rect=None):
        """
        Plot the n > n + 1 map of the variable mapped.

        Parameters
        ----------
        ax: ``matplotlib.pyplot.Axis``, (optional)
            Matplotlib axis specification.
        kws:```dict``  (optional)
            The parameters to send to set up the plot, like line colors,
            linewidth, marker face color, etc.
            Here is a list of available line properties:
            matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html


        Returns
        -------
        `AxesSubplot``
            A configuration representing the axis data.
        """
        ax = plt.gca() if ax is None else ax

        kws_scatter = {} if kws_scatter is None else kws_scatter

        kws_rect = {} if kws_rect is None else kws_rect

        kws_scatter.setdefault("color", "r")
        kws_scatter.setdefault("s", [2])

        kws_rect.setdefault("color", "k")

        n0 = self.n0[self.n0 != 0]
        n1 = self.n1[self.n1 != 0]
        ax.scatter(n0, n1, **kws_scatter)
        min_n0 = min(n0)
        max_n0 = max(n0)
        min_n1 = min(n1)
        max_n1 = max(n1)
        ax.plot(
            np.linspace(min_n0, max_n0, 100),
            np.linspace(min_n1, max_n1, 100),
            **kws_rect,
        )
#        ax.set_xlim([min_n0, max_n0])
#        ax.set_ylim([min_n1, max_n1])
        ax.ticklabel_format(axis="both", style="plain", useOffset=False)
        return ax

    def plot_iterations(self, ax=None, kws=None):
        """
        Plot the axis variable versus the time of intersection.

        Parameters
        ----------
        ax: ``matplotlib.pyplot.Axis``, (optional)
            Matplotlib axis specification.
        kws:```dict``  (optional)
            The parameters to send to set up the plot, like line colors,
            linewidth, marker face color, etc.
            Here is a list of available line properties:
            matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html


        Returns
        -------
        `AxesSubplot``
            A configuration representing the axis data.

        """
        ax = plt.gca() if ax is None else ax

        kws = {} if kws is None else kws

        kws.setdefault("color", "k")
        kws.setdefault("marker", ".")

        ax.plot(self.t_iter, self.iterations, **kws)

        return ax
