# from matplotlib import pyplot as plt
# from sympy import *

# ==============================================================================
# Docs
# ==============================================================================

"""Stability and temporal analysis of dynamical systems."""

# ==============================================================================
# Imports
# ==============================================================================

import numpy as np

import pandas as pd

from scipy.integrate import solve_ivp

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

from . import poincare, trajectories

# ==============================================================================
# Class Functional
# ==============================================================================


class Functional:
    """Dynamical system defined by its derivatives and name.

    Callable type should take a state vector, a time value and a parameters
    vector and then must compute and return the derivative of the system in
    respect to a given state of it.

    dy / dt = f(x, t, p)

    Provides access to the attribute of the function that gives the derivatives
    for each variable, the name is also an attribute together with the
    variables in case these are defined.

    Parameters
    ----------
    func: callable
        The func is used to compute the derivative of the system given a set of
         variable values, and parameters.
    name: str
        System's name.
    *args
        The variable arguments are the variable names, and sometimes are needed
        for implementing methods in subclasses.

    Attributes
    ----------
    func: callable
        This is where we store func.
    name: str
        This is where we store name.
    variables: list, optional (default=[x_1, ..., x_n])
        System's variable list.

    Example
    -------
    >>> def sample_function(x, t, p):
        x1, x2 = x
        p1, p2 = p
        dx1 = x1**2 + p1 * x2
        dx2 = -p2 * x2 + x1
        return [dx1, dx2]

    >>> name = 'Sample System'
    >>> sample_sys = Functional(sample_function, name)
    >>> sample_sys.func
    <function __main__.sample_function()>

    >>> sample_sys.name
    'Sample System'
    """

    def __init__(self, func, name, *variables):
        if not callable(func):
            raise TypeError(
                "The first argument must be a callable"
                + f"got {type(func)} instead."
            )
        if not isinstance(name, str):
            raise TypeError(
                "The second argument must be a string"
                + f"got {type(name)} instead."
            )
        self.func = func
        self.name = name
        if variables == ():
            variables = [[]]

        self.variables = variables[0]

    def time_evolution(
        self,
        x0,
        parameters,
        ti=0,
        tf=200,
        n=5000,
        rel_tol=1e-10,
        abs_tol=1e-12,
        mx_step=0.004,
    ):
        """
        Integrates a system in forward time.

        Parameters
        ----------
        x0: ``list``
            Set of initial conditions of the system. Values inside must
            be of int or float type.
        parameters: ``list``
            Set of the function's parameter values for this particular case.
        ti: ``int, float``, optional (default=0)
            Initial integration time value.
        tf: ``int, float``, optional (default=200)
            Final integration time value.
        rel_tol: ``float``, optional (default=1e-10)
            Relative tolerance of integrator.
        abs_tol:``float``, optional (default=1e-12)
            Absolute tolerance of integrator.
        mx_step: ``int, float``, optional (default=0.004)
            Maximum integration step of integrator.


        Raises
        ------
        ValueError
            Final integration time must be greater than initial
            integration time.


        Returns
        -------
        Trajectory: caospy.trajectories.Trajector
            Trajectory of a dynamical system in a given time interval.


        See Also
        --------
        scipy.integrate.solve_ivp
        caospy.trajectories.Trajectory


        Example
        -------
        >>> sample_sys = Functional(sample_function, name)
        >>> sample_x0 = [1, 0.5]
        >>> sample_p = [2, 4]
        >>> t1 = sample_sys.time_evolution(sample_x0, sample_p)
        >>> t1
        <caospy.trajectories.Trajectory at 0x18134df0>
        >>> type(t1)
        caospy.trajectories.Trajectory

        """
        if not tf > ti:
            raise ValueError(
                "Final integration time must be"
                + "greater than initial integration time."
            )
        if int((tf - ti) / n) > mx_step:
            n = int((tf - ti) / mx_step)

        t_ev = np.linspace(ti, tf, n)
        sol = solve_ivp(
            self.func,
            [ti, tf],
            x0,
            args=(parameters,),
            t_eval=t_ev,
            rtol=rel_tol,
            atol=abs_tol,
            max_step=mx_step,
            dense_output=True,
        )
        x = sol.sol(t_ev)
        return trajectories.Trajectory(t_ev, x, self.variables)

    def poincare(
        self,
        x0,
        parameters,
        t_disc=5000,
        t_calc=50,
        n=2000,
        rel_tol=1e-10,
        abs_tol=1e-12,
        mx_step=0.004,
    ):
        """Integrates a system forward in time, eliminating the transient.

        Then returns a Poincare type object, which can be worked with to
        get the Poincaré maps.


        Parameters
        ----------
        x0: list
            Set of initial conditions of the system. Values inside must be
            of int or float type.
        parameters: list
            Set of parameter values for this particular case.
        t_disc: int, optional (default=5000)
            Transient integration time to be discarded next.
        t_calc: int, optional (default=50)
            Stationary integration time, the system's states corresponding
            to this integration time interval are kept and pass to the
            Poincare object.
        rel_tol: float, optional (default=1e-10)
            Relative tolerance of integrator.
        abs_tol:float, optional (default=1e-12)
            Absolute tolerance of integrator.
        mx_step: int, float, optional(default=0.01)
            Maximum integration step of integrator.


        Returns
        -------
        Poincare: caospy.poincare.Poincare
            Poincare object defined by t_calc time vector and matrix of states.


        See Also
        --------
        caospy.trajectories.Trajectory


        Example
        -------
        >>> sample_sys = Functional(sample_function, name)
        >>> sample_x0 = [1, 0.5]
        >>> sample_p = [2, 4]
        >>> t_desc = 10000
        >>> t_calc = 45
        >>> p1 = sample_sys.poincare(sample_x0, sample_p, t_desc, t_calc)
        >>> p1
        <caospy.poincare.Poincare at 0x18d91028>
        >>> type(p1)
        caospy.poincare.Poincare

        """
        # Integrate for the discard time, to eliminate the transient.
        if int(t_disc / n) > mx_step:
            n1 = int(t_disc / mx_step)
        else:
            n1 = n

        if int(t_calc / n) > mx_step:
            n2 = int(t_calc / mx_step)
        else:
            n2 = n

        sol_1 = self.time_evolution(
            x0, parameters, 0, t_disc, n1, rel_tol, abs_tol, mx_step
        )
        x1 = sol_1.x
        # Then get the stationary trajectory.
        x0_2 = x1[:, -1]
        sol_2 = self.time_evolution(
            x0_2, parameters, 0, t_calc, n2, rel_tol, abs_tol, mx_step
        )
        t, x = sol_2.t, sol_2.x
        return poincare.Poincare(t, x, self.variables)


# ==========================================================================
# Class Symbolic
# ==========================================================================


class Symbolic(Functional):
    """
    Dynamical system defined by variables, parameters and functions.

    Variables, functions and parameters must be lists of strings, the number
    of variables must match the number of equations, and the name should
    be a string.

    The available attributes are the inputed variables, functions,
    parameters and name just as they were given. And "privately" defined,
    are the variables and parameters dict, which will store the
    sympy.symbols for the parameters and variables, and lastly the
    sympy.Equations list containing the functions.


    Parameters
    ----------
    x: list
        System's list of variable names.
    f: list
        System's list of string functions.
    params: list
        System's list of parameter names.
    name: str
        System's name.


    Attributes
    ----------
    _name: str

    f: list
        Here we store the f argument.
    x: list
        Here we store the x argument.
    params: list
        Here we store the params argument.
    _variables: dict
        Dictionary with variables stored with variable name of str type as keys
        and variables defined as sympy.Symbol as values.
    _parameters: dict
        Dictionary with parameters stored with variable name of str type as
        keys and parameters defined as sympy.Symbol as values.
    _equations: list
        List with system's functions stored as sympy.Equations, all equated
        to 0.


    Raises
    ------
    TypeError
        Name must be a string, got {type(name)} instead.
    TypeError
        The variables, functions and parameters should be lists,
        got {(type(x), type(f), type(params))} instead.
    TypeError
        All the elements must be strings.
    ValueError
        System must have equal number of variables and equations,
        instead has {len(x)} variables"and {len(f)} equations


    Example
    -------
    >>> v = ['x1', 'x2', 'x3']
    >>> f = ['x1 * x2 - a', '-x3 + b * (x2**2)', 'c * x1 / x2']
    >>> p = ['a', 'b', 'c']
    >>> sample_symbolic = caospy.Symbolic(v, f, p, 'sample_sys')
    >>> sample_symbolic
    <caospy.core.Symbolic object at 0x000001A57CF094C0>

    We can get the __init__ attributes as they were plugged:

    >>> sample_symbolic.x
    ['x1', 'x2', 'x3']
    >>> sample_symbolic.f
    ['x1 * x2 - a', '-x3 + b * (x2**2)', 'c * x1 / x2']
    >>> sample_symbolic.params
    ['a', 'b', 'c']

    In order to work with the sympy library, the arguments are adapted
    into sympy types and stored in different "private" attributes.

    >>> sample_symbolic._name
    'sample_sys'
    >>> sample_symbolic._variables
    {'x1': x1, 'x2': x2, 'x3': x3}
    >>> sample_symbolic._parameters
    {'a': a, 'b': b, 'c': c}
    >>> sample_symbolic._equations
    [Eq(-a + x1*x2, 0), Eq(b*x2**2 - x3, 0), Eq(c*x1/x2, 0)]

    """

    def __init__(self, x, f, params, name):
        # Making sure the arguments are of the adequate form and type.
        if not isinstance(name, str):
            raise TypeError(
                f"Name must be a string, got {type(name)} instead."
            )
        if not all(isinstance(i, list) for i in [x, f, params]):
            raise TypeError(
                "The variables, functions and parameters"
                + "should be lists, got"
                + f"{(type(x), type(f), type(params))} instead."
            )
        for i in [x, f, params]:
            if not all(isinstance(j, str) for j in i):
                raise TypeError("All the elements must be strings.")
        if not len(x) == len(f):
            raise ValueError(
                "System must have equal number of variables"
                + f"and equations, instead has {len(x)} variables"
                + f"and {len(f)} equations"
            )

        self._name = name
        self.f = f
        self.x = x
        self.params = params
        self._variables = {}
        self._parameters = {}
        self._equations = []

        # Making variable's strings into sympy symbols.
        for v in x:
            if not isinstance(v, sp.Symbol):
                v = sp.symbols(v)
            self._variables[v.name] = v

        # Making parameters's strings into sympy symbols.
        for p in params:
            if not isinstance(p, sp.Symbol):
                p = sp.symbols(p)
            self._parameters[p.name] = p

        # Making function's strings into sympy equations.
        local = {**self._variables, **self._parameters}
        for eq in self.f:
            if not isinstance(eq, sp.Eq):
                eq = sp.Eq(parse_expr(eq, local.copy(), {}), 0)
            self._equations.append(eq)

        # Creating function from sympy equations.
        function = []
        for i in range(len(self._equations)):
            try:
                function.append(self._equations[i].args[0])
            except IndexError:
                function.append(parse_expr(self.f[i], local.copy(), {}))

        dydt = sp.lambdify(([*self._variables], [*self._parameters]), function)

        def fun(t_fun, x_fun, par_fun):
            return dydt(x_fun, par_fun)

        v_names = list(self._variables.keys())
        super().__init__(fun, self._name, v_names)

    def _linear_analysis(self, p, initial_guess, reach=5):
        """
        Compute the system's roots.

        They're called "fixed points" in dynamical systems,
        given that the derivative is zero in them. Then it also
        gets the Jacobian matrix for the system and evaluates it in the
        different roots to find the eigenvalues and eigenvectors.


        Parameters
        ----------
        p: list, tuple
            Set of parameters values that will specify the system.
        initial_guess: list, tuple
            If the function is not implemented by the sympy.solve solver,
            then it won't return all the system's roots, but will return
            the single closest root to the guessed value.
        reach: int, optional (default=5)
            Multistate flag variable that will dictate how far the method
            should go into computing the different elements needed for the
            linear stability analysis. If 1, it will only return the roots,
            if 2, will return only the evaluated jacobians, if 3 it returns
            the eigenvalues, if 4 returns only eigenvectors, and finally if 5,
            it returns all of the previous.


        Returns
        -------
        list
            List containing roots, evaluated jacobians, eigenvalues and
            eigenvectors in that order. Output could be captured by separated
            variables.


        Examples
        --------
        Initialize a Symbolic type object.

        >>> v = ['x1', 'x2', 'x3']
        >>> f = ['x1 * x2 - a', '-x3 + b * (x2**2)', 'c * x1 - x2']
        >>> p = ['a', 'b', 'c']
        >>> sample_symbolic = caospy.Symbolic(v, f, p, 'sample_sys')

        Define the values of the parameters, and the initial guess, in case
        that sympy.solve can't find the roots, and sympy.nsolve needs to be
        implemented.

        >>> p_values = [1, 1, 1]
        >>> initial_guess = []
        >>> sample_symbolic._linear_analysis(p_values, initial_guess, 1)
        [array([[-1., -1.,  1.],
               [ 1.,  1.,  1.]]), None, None, None]

        >>> sample_symbolic._linear_analysis(p_values, initial_guess, 2)
        [None, array([[[-1., -1.,  0.],
                [ 0., -2., -1.],
                [ 1., -1.,  0.]],

               [[ 1.,  1.,  0.],
                [ 0.,  2., -1.],
                [ 1., -1.,  0.]]]), None, None]

        >>> sample_symbolic._linear_analysis(p_values, initial_guess, 3)
        [None, None, array([[ 0.61803399, -1.61803399, -2.        ],
               [-0.61803399,  1.61803399,  2.        ]]), None]

        >>> sample_symbolic._linear_analysis(p_values, initial_guess, 4)
        [None, None, None,
        array([[ 2.15353730e-01, -3.48449655e-01,  9.12253040e-01],
               [ 8.34001352e-01,  5.15441182e-01, -1.96881012e-01],
               [-7.07106781e-01, -7.07106781e-01, -1.76271580e-16]])]

        [array([[-1., -1.,  1.],
               [ 1.,  1.,  1.]]), array([[[-1., -1.,  0.],
                [ 0., -2., -1.],
                [ 1., -1.,  0.]],

               [[ 1.,  1.,  0.],
                [ 0.,  2., -1.],
                [ 1., -1.,  0.]]]), array([[ 0.61803399, -1.61803399, -2.],
               [-0.61803399,  1.61803399,  2.]]),
               array([[ 2.15353730e-01, -3.48449655e-01,  9.12253040e-01],
               [ 8.34001352e-01,  5.15441182e-01, -1.96881012e-01],
               [-7.07106781e-01, -7.07106781e-01, -1.76271580e-16]])]

        """
        if len(initial_guess) == 0:
            initial_guess = [0 for i in self._variables.values()]

        parameter_list = list(self._parameters.values())
        replace = list(zip(parameter_list, p))
        equalities = [eqn.subs(replace) for eqn in self._equations]
        # Sometimes the solve method works fine, but sometimes nsolve
        # is needed.
        try:
            roots = sp.solve(equalities, list(self._variables.values()))
        except NotImplementedError:
            try:
                roots = [
                    tuple(
                        sp.nsolve(
                            equalities,
                            list(self._variables.values()),
                            initial_guess,
                        )
                    )
                ]
            except TypeError:
                raise TypeError(
                    "Initial guess is not allowed,"
                    + "try with another set of values"
                )
        if len(roots) == 0:
            return [None, None, None, None]

        # Sometimes solve returns a dict, which has to be
        # converted into a list to handle it like all the
        # other results.
        if isinstance(roots, dict):
            var_values = list(self._variables.values())
            roots_keys = list(roots.keys())
            if var_values != roots_keys:
                for k in var_values:
                    if k not in roots_keys:
                        roots[k] = k

            roots = [tuple(roots.values())]

        # If elements are sympy symbols, we cannot convert
        # them into floats nor complexes, so we just let
        # them like they are.
        for i in range(len(roots)):
            try:
                roots[i] = list(map(float, roots[i]))
            except TypeError:
                try:
                    roots[i] = list(map(complex, roots[i]))
                except TypeError:
                    roots[i] = list(roots[i])

        roots = np.array(roots)
        if reach == 1:
            return [roots, None, None, None]

        expresions = [eqn.args[0] for eqn in self._equations]
        equations = [exp.subs(replace) for exp in expresions]
        jacobian = np.array(
            [[sp.diff(eq, var) for var in self._variables] for eq in equations]
        )

        variable_list = list(self._variables.values())
        replace_values = [list(root) for root in roots]
        replace = [list(zip(variable_list, i)) for i in replace_values]
        n_var = len(self._variables)
        n_eq = len(self._equations)
        n_roots = len(replace)
        a_matrices = np.zeros((n_roots, n_eq, n_var), dtype=object)
        for i, rep in enumerate(replace):
            for j, row in enumerate(jacobian):
                for k, derivative in enumerate(row):
                    a_matrices[i, j, k] = derivative.subs(rep)

        try:
            a_matrices = a_matrices.astype("float64")
        except TypeError:
            try:
                a_matrices = a_matrices.astype("complex128")
            except TypeError:
                pass

        if reach == 2:
            return [None, a_matrices, None, None]

        elif reach == 3:
            w, v = np.linalg.eig(a_matrices)
            return [None, None, w, None]

        elif reach == 4:
            w, v = np.linalg.eig(a_matrices)
            v = np.array([i.T for i in v])
            return [None, None, None, v]
        else:
            w, v = np.linalg.eig(a_matrices)
            v = np.array([i.T for i in v])
            return [roots, a_matrices, w, v]

    def fixed_points(self, p, initial_guess=[]):
        """
        Return the roots of the system, given a set of parameters values.

        If function is not implemented by sypmpy.solve, a set of initial
        guess values is needed.


        Parameters
        ----------
        p: ``list``, ``tuple``
            Set of parameter values, they should be int or float type.
        initial_guess: ``list``, ``tuple``, optional (default=[0, ..., 0])


        Return
        ------
        out: np.array
            Numpy array containing one row per root, and one column per
            variable.


        Example
        -------
        >>> v = ['x1', 'x2', 'x3']
        >>> f = ['x1 * x2 - a', '-x3 + b * (x2**2)', 'c * x1 - x2']
        >>> p = ['a', 'b', 'c']
        >>> p_values = [1, 1, 1]
        >>> sample_symbolic.fixed_points(p_values)
        array([[-1., -1.,  1.],
               [ 1.,  1.,  1.]])

        Redefining the parameter values

        >>> p_values = [-1, 3, 5]
        >>> sample_symbolic.fixed_points(p_values)
        array([[  0.-0.4472136j ,   0.-2.23606798j, -15.+0.j],
               [  0.+0.4472136j ,   0.+2.23606798j, -15.+0.j]])

        """
        return self._linear_analysis(p, initial_guess, 1)[0]


# ==========================================================================
# Class MultiVarMixin
# ==========================================================================


class MultiVarMixin(Symbolic):
    """Multivariable system's specific implementations."""

    def jacob_eval(self, p, initial_guess=[]):
        """
        Compute the evaluated  fixed points Jacobian matrix.

        Parameters
        ----------
        p: ``list``, ``tuple``
            Set of parameter values, they should be int or float type.
        initial_guess: ``list``, ``tuple``, optional (default=[0, ..., 0])


        Return
        ------
        out: array
            Numpy array, of shape (i, j, k), where i is the number of
            fixed points, j is the number of equations of the system, and
            k the number of variables. For design reasons j=k.

            The element i, j, k is the derivative of the function j
            respect to the variable k evaluated in the ith fixed point.

        Example
        -------
        In order to implement the method, we initialize a MultiDim type
        object, see ``MultiDim`` class to know this implementation.

        >>> v = ['x1', 'x2', 'x3']
        >>> f = ['x1 * x2 - a', '-x3 + b * (x2**2)', 'c * x1 - x2']
        >>> p = ['a', 'b', 'c']
        >>> p_values = [-1, 3, 5]
        >>> sample_multidim = caospy.MultiDim(v, f, p, 'sample_sys')
        >>> sample_multidim.jacob_eval(p_values)
        array([[[ 0. -2.23606798j,  0. -0.4472136j ,  0. +0.j        ],
                [ 0. +0.j        ,  0.-13.41640786j, -1. +0.j        ],
                [ 5. +0.j        , -1. +0.j        ,  0. +0.j        ]],

               [[ 0. +2.23606798j,  0. +0.4472136j ,  0. +0.j        ],
                [ 0. +0.j        ,  0.+13.41640786j, -1. +0.j        ],
                [ 5. +0.j        , -1. +0.j        ,  0. +0.j        ]]])


        """
        return self._linear_analysis(p, initial_guess, 2)[1]

    def eigenvalues(self, p, initial_guess=[]):
        """
        Compute the eigenvalues, of all the system's fixed points.

        Parameters
        ----------
        p: ``list``, ``tuple``
            Set of parameter values, they should be int or float type.
        initial_guess: ``list``, ``tuple``,
        optional (default=[0, ..., 0])


        Return
        ------
        out: array
            Numpy array, of shape (i, j), where i is the number of
            fixed points, j is the number of variables.


        Example
        -------
        In order to implement the method, we initialize a MultiDim
        type object, see ``MultiDim`` class to know this implementation.

        >>> v = ['x1', 'x2', 'x3']
        >>> f = ['x1 * x2 - a', '-x3 + b * (x2**2)', 'c * x1 - x2']
        >>> p = ['a', 'b', 'c']
        >>> sample_multidim = caospy.MultiDim(v, f, p, 'sample_sys')
        >>> p_values = [1, 1, 1]
        >>> sample_multidim.eigenvalues(p_values)
        array([[ 0.61803399, -1.61803399, -2.        ],
               [-0.61803399,  1.61803399,  2.        ]])


        """
        return self._linear_analysis(p, initial_guess, 3)[2]

    def eigenvectors(self, p, initial_guess=[]):
        """
        Compute the eigenvectors, of all the system's fixed points.

        Parameters
        ----------
        p: ``list``, ``tuple``
            Set of parameter values, they should be int or float type.
        initial_guess: ``list``, ``tuple``, optional (default=[0, ..., 0])


        Return
        ------
        out: array
            Numpy array, of shape (i, j, k), where i is the number of
            fixed points, j is the number of variables.

            The element i, j, k is the component in the kth direction,
            of the jth vector, of the ith root.


        Example
        -------

        In order to implement the method, we initialize a MultiDim type
        object, see ``MultiDim`` class to know this implementation.

        >>> v = ['x1', 'x2', 'x3']
        >>> f = ['x1 * x2 - a', '-x3 + b * (x2**2)', 'c * x1 - x2']
        >>> p = ['a', 'b', 'c']
        >>> sample_multidim = caospy.MultiDim(v, f, p, 'sample_sys')
        >>> p_values = [1, 1, 1]
        >>> sample_multidim.eigenvectors(p_values)
        array([[[ 2.15353730e-01, -3.48449655e-01,  9.12253040e-01],
            [ 8.34001352e-01,  5.15441182e-01, -1.96881012e-01],
            [-7.07106781e-01, -7.07106781e-01, -1.76271580e-16]],

           [[-2.15353730e-01,  3.48449655e-01,  9.12253040e-01],
            [ 8.34001352e-01,  5.15441182e-01,  1.96881012e-01],
            [-7.07106781e-01, -7.07106781e-01,  1.76271580e-16]]])

        """
        return self._linear_analysis(p, initial_guess, 4)[3]

    def full_linearize(self, p, initial_guess=[]):
        """
        Compute the roots, evaluated jacobians, eigenvalues and eigenvectors.

        Parameters
        ----------
        p: ``list``, ``tuple``
            Set of parameter values, they should be int or float type.
        initial_guess: ``list``, ``tuple``, optional (default=[0, ..., 0])


        Return
        ------
        out: list
            List containing the roots as its first element, the evaluated
            jacobians as the second, the eigenvalues as third and finally
            the eigenvectors. The type and shape of each are the same as
            in their particular implementations. See ``fixed_points``,
            ``jacob_eval``, ``eigenvalues`` and ``eigenvectors``
            for further detail.


        Example
        -------

        In order to implement the method, we initialize a MultiDim type object,
        see ``MultiDim`` class to know this implementation.

        >>> v = ['x1', 'x2', 'x3']
        >>> f = ['x1 * x2 - a', '-x3 + b * (x2**2)', 'c * x1 - x2']
        >>> p = ['a', 'b', 'c']
        >>> sample_multidim = caospy.MultiDim(v, f, p, 'sample_sys')
        >>> p_values = [1, 1, 1]
        >>> sample_symbolic.full_linearize(p_values)
               [array([[-1., -1.,  1.],
               [ 1.,  1.,  1.]]), array([[[-1., -1.,  0.],
                [ 0., -2., -1.],
                [ 1., -1.,  0.]],
               [[ 1.,  1.,  0.],
                [ 0.,  2., -1.],
                [ 1., -1.,  0.]]]), array([[ 0.61803399, -1.61803399, -2.],
               [-0.61803399,  1.61803399,  2.        ]]),
               array([[[ 2.15353730e-01, -3.48449655e-01,  9.12253040e-01],
                [ 8.34001352e-01,  5.15441182e-01, -1.96881012e-01],
                [-7.07106781e-01, -7.07106781e-01, -1.76271580e-16]],
               [[-2.15353730e-01,  3.48449655e-01,  9.12253040e-01],
                [ 8.34001352e-01,  5.15441182e-01,  1.96881012e-01],
                [-7.07106781e-01, -7.07106781e-01,  1.76271580e-16]]])]

        """
        return self._linear_analysis(p, initial_guess, 5)


# ==========================================================================
# Class OneDimMixin
# ==========================================================================


class OneDimMixin(Symbolic):
    """Specific behaviors of onedimensional systems."""

    def stability(self, parameters):
        """
        Compute the slope of the derivative function at the fixed points.

        It gives you the system's stability


        Parameters
        ----------
        parameters: ``list``, ``tuple``
            Set of values of the parameters, that specify the system.


        Return
        ------
        out: pd.DataFrame
            Pandas data frame, which columns are "Fixed point", "Slope",
            "Stability". It has a row for every fixed point of the system.


        Example
        -------
        >>> v = ['x1']
        >>> f = ['a * x1**2 + b * x1 + c']
        >>> p = ['a', 'b', 'c']
        >>> sample_onedim = caospy.OneDim(v, f, p, 'sample_1d')
        >>> p_values = [1, 1, -4]
        >>> sample_onedim.stability(p_values)
                     Fixed Point     Slope  Stability
        0   [1.5615528128088303]  4.123106      False
        1  [-2.5615528128088303] -4.123106       True

        """
        replace_params = list(zip(self._parameters.values(), parameters))
        equation = self._equations[0].args[0].subs(replace_params)
        derivative = sp.diff(equation, list(self._variables.values())[0])
        zero = self.fixed_points(parameters)
        if zero is None:
            return "There are no fixed points to evaluate"
        replace_variables = []
        for z in zero:
            replace_variables.append(list(zip(self._variables, z)))

        slopes = []
        for value in replace_variables:
            slopes.append(float(derivative.subs(value)))

        for i in range(len(zero)):
            zero[i] = zero[i][0]

        stable = []
        for slope in slopes:
            if slope == 0:
                raise LinearityError(
                    "Linear stability is undefined."
                    " Slope in fixed point is 0."
                )

            stable.append(True if slope < 0 else False)
        data = pd.DataFrame(
            list(zip(zero, slopes, stable)),
            columns=["Fixed Point", "Slope", "Stability"],
        )
        return data


# ==========================================================================
# Class OneDim
# ==========================================================================


class OneDim(OneDimMixin, Symbolic):
    """
    Captures the specific tools of analysis for onedimensional systems.

    Has the same attributes as the Symbolic class.


    Example
    -------
    >>> v = ['x1']
    >>> f = ['a * x1**2 + b * x1 + c']
    >>> p = ['a', 'b', 'c']
    >>> sample_onedim = caospy.OneDim(v, f, p, 'sample_1d')
    >>> sample_onedim
    <caospy.core.OneDim object at 0x0000027546C36850>

    """

    def __init__(self, x, f, params, name):
        if not (len(x) == 1 & len(f) == 1):
            raise Exception(
                f"""System shape is {len(x)} by {len(f)} but it should be 1
                    by 1"""
            )
        super().__init__(x, f, params, name)


# ==========================================================================
# Class TwoDimMixin
# ==========================================================================


class TwoDimMixin(MultiVarMixin, Symbolic):
    """Specific behaviors of twodimensional systems."""

    def fixed_point_classify(self, params_values, initial_guess=[]):
        """
        Fix points classification in 2D.

        Classifies the fixed points according to their linear stability,
        based on the values of the trace and determinant given by the evaluated
        jacobian matrix.

        Parameters
        ----------
        params_values: ``list``,``tuple``
            Set of specific parameter values to fix the system.
        initial_guess: ``list``, ``tuple``, optional (default=[0, ..., 0])


        Return
        ------
        out: DataFrame
            Pandas data frame with a row for every fixed point,
            and columns being:
            "var1", "var2", "λ_{1}$", "$λ_{2}$", "$σ$", "$Δ$", "$Type$".

            The first two columns have the values of the variables where
            the fixed point is. Then the two eigenvalues, the trace and
            determinant, and finally the classification of the fixed point.


        Examples
        --------
        >>> variables = ["x", "y"]
        >>> functions = ["x + a * y", "b * x + c * y"]
        >>> parameters = ["a", "b", "c"]
        >>> sample_TwoDim = caospy.TwoDim(variables, functions, parameters,
                                          "sample2d")
        >>> p_values = [2, 3, 4]
        >>> sample_TwoDim.fixed_point_classify(p_values)
           $x$  $y$ $λ_{1}$ $λ_{2}$     $σ$      $Δ$  $Type$
        0  0.0  0.0   -0.37    5.37  (5+0j)  (-2+0j)  Saddle

        >>> functions = ["a * y", "-b * x - c * y"]
        >>> p_values = [1, -2, 3]
        >>> sample_TwoDim = caospy.TwoDim(variables, functions, parameters,
                                          'sample2d')
        >>> sample_TwoDim.fixed_point_classify(p_values)
           $x$  $y$ $λ_{1}$ $λ_{2}$      $σ$     $Δ$       $Type$
        0  0.0  0.0    -1.0    -2.0  (-3+0j)  (2+0j)  Stable Node

        """
        a = self.jacob_eval(params_values, initial_guess)
        roots = self.fixed_points(params_values, initial_guess)
        if a is None:

            return "There is no fixed points to evaluate."

        traces = []
        dets = []
        classification = []
        for i, r in enumerate(roots):
            # Calculate trace and determinant.
            if len(r) == 1:
                trace = a[0][0] + a[1][1]
                det = a[0][0] * a[1][1] - a[1][0] * a[0][1]
            else:
                trace = a[i][0][0] + a[i][1][1]
                det = a[i][0][0] * a[i][1][1] - a[i][1][0] * a[i][0][1]

            traces.append(np.around(complex(trace), 2))
            dets.append(np.around(complex(det), 2))
            # Classify fixed point according to trace and det.
            if det == 0:
                if trace < 0:
                    classification.append(
                        "Non Isolated Fixed-Points,"
                        + "Line of Lyapunov stable fixed points"
                    )
                elif trace == 0:
                    classification.append(
                        "Non Isolated Fixed-Points," + "Plane of fixed points"
                    )
                elif trace > 0:
                    classification.append(
                        "Non Isolated Fixed-Points,"
                        + "Line of unstable fixed points."
                    )

            elif det < 0:
                classification.append("Saddle")

            elif det > 0:
                if trace == 0:
                    classification.append("Center")

                if trace ** 2 - 4 * det > 0:
                    if trace > 0:
                        classification.append("Unstable Node")
                    elif trace < 0:
                        classification.append("Stable Node")

                elif trace ** 2 - 4 * det < 0:
                    if trace > 0:
                        classification.append("Unstable Spiral")
                    elif trace < 0:
                        classification.append("Stable Spiral")

                elif trace ** 2 - 4 * det == 0:
                    if a[i][0][1] == 0 and a[i][1][0] == 0:
                        if trace > 0:
                            classification.append("Unstable Star Node")
                        elif trace < 0:
                            classification.append("Stable Star Node")

                    else:
                        if trace > 0:
                            classification.append("Unstable Degenerate Node")
                        elif trace < 0:
                            classification.append("Stable Degenerate Node")

        roots = self.fixed_points(params_values, initial_guess)
        eigen = self.eigenvalues(params_values, initial_guess)
        traces = np.array(traces)
        dets = np.array(dets)
        classification = np.array(classification)
        data_array = np.empty((roots.shape[0], 7), dtype=object)
        for i in range(roots.shape[0]):
            data_array[i][0] = roots[i][0]
            data_array[i][1] = roots[i][1]
            data_array[i][2] = eigen[i][0]
            data_array[i][3] = eigen[i][1]
            data_array[i][4] = traces[i]
            data_array[i][5] = dets[i]
            data_array[i][6] = classification[i]

        cols = [f"${v}$" for v in list(self._variables.values())] + [
            "$\u03BB_{1}$",
            "$\u03BB_{2}$",
            "$\u03C3$",
            "$\u0394$",
            "$Type$",
        ]
        pd.set_option("display.precision", 2)
        data = pd.DataFrame(data_array, columns=cols)
        return data


# ==========================================================================
# Class TwoDim
# ==========================================================================


class TwoDim(TwoDimMixin, Symbolic):
    """
    Specific implementations 2D.

    Englobes the specific implementations of the tools of analysis for systems.


    Example
    -------
    >>> variables = ["x", "y"]
    >>> functions = ["x + a * y", "b * x + c * y"]
    >>> parameters = ["a", "b", "c"]
    >>> sample_TwoDim = caospy.TwoDim(variables, functions, parameters,
        "sample2d")
    >>> sample_TwoDim
    <caospy.core.TwoDim object at 0x0000027561643A60>

    """

    def __init__(self, x, f, params, name):
        if not (len(x) == 2 & len(f) == 2):
            raise ValueError(
                f"System shape is {len(x)} by"
                + f"{len(f)} but it should be 2 by 2"
            )
        super().__init__(x, f, params, name)


# ==========================================================================
# Class MultiDim
# ==========================================================================


class MultiDim(MultiVarMixin, Symbolic):
    """
    Multidimensional systems.

    Implements the specific stability analysis tools and behaviorfor
    the multidimensional systems.

    Example
    -------
    >>> variables = ["x", "y"]
    >>> functions = ["x+a*y", "b*x+c*y"]
    >>> parameters = ["a", "b", "c"]
    >>> sample_MultiDim = caospy.MultiDim(variables, functions, parameters,
                                          'sampleMultiDim')
    >>> sample_MultiDim
    <caospy.core.MultiDim object at 0x0000027861643A60>
    """

    def __init__(self, x, f, params, name):
        super().__init__(x, f, params, name)


# ==========================================================================
# Class AutoSymbolic
# ==========================================================================


class AutoSymbolic(Symbolic):
    """Initializes predetermined dynamical system."""

    def __init__(self):
        # Initializes systems that are predefined in the code.
        cls = type(self)
        super().__init__(
            x=cls._variables,
            f=cls._functions,
            params=cls._parameters,
            name=cls._name,
        )


# ==========================================================================
# Class LinearityError
# ==========================================================================


class LinearityError(ValueError):
    """Exception indicating unmatching number of equations and variables."""

    pass
