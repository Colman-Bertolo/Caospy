# import matplotlib.pyplot as plt
# import pandas as pd
# from sympy import *

import types
import numpy as np
import sympy as sp
from scipy.integrate import odeint
from sympy.parsing.sympy_parser import parse_expr


class Functional:
    def __init__(self, func, name):
        assert isinstance(func, types.FunctionType), f"The first argument must be of the type 'Function', got {type(func)} instead."
        assert isinstance(name, str), f"The second argument must be of the type 'string' got {type(name)} instead."

        self.func = func
        self.name = name

    def time_evolution(self, x0, ti, tf, n, parameters):
        assert tf > ti, 'Final integration time must be greater than initial integration time.'
        t = np.linspace(ti, tf, int(n))
        y = odeint(self.func, x0, t, args=(parameters,))
        return [t, y]


class Symbolic(Functional):
    def __init__(self, x, f, params, name):
        assert isinstance(name, str), f'Name must be a string, got {type(name)} instead.'
        assert all(isinstance(i, list) for i in [x, f, params]), f'The variables, functions and parameters should be entered as lists, got {(type(x), type(f), type(params))} instead.'
        for i in [x, f, params]:
            assert all(isinstance(j, str) for j in i), f'All the elements must be strings.'


        self._name = name
        self.f = f
        self._variables = {}
        self._parameters = {}
        self._equations = []

        for v in x:
            if not isinstance(v, sp.Symbol):
                v = sp.symbols(v)
            self._variables[v.name] = v

        for p in params:
            if not isinstance(p, sp.Symbol):
                p = sp.symbols(p)
            self._parameters[p.name] = p

        local = {**self._variables, **self._parameters}
        for eq in self.f:
            if not isinstance(eq, sp.Eq):
                eq = sp.Eq(parse_expr(eq, local.copy(), {}), 0)
            self._equations.append(eq)
        function = []
        for eq in self._equations:
            function.append(eq.args[0])
        dydt = sp.lambdify(([*self._variables], [*self._parameters]), function)

        def fun(x_fun, t_fun, par_fun):
            return dydt(x_fun, par_fun)

        super().__init__(fun, self._name)

    def _linear_analysis(self, p, reach=3):
        parameter_list = list(self._parameters.values())
        replace = list(zip(parameter_list, p))
        equalities = [eqn.subs(replace) for eqn in self._equations]
        roots = sp.solve(equalities, self._variables)
        roots = [list(i.values()) for i in roots]
        if reach == 1:
            return roots
        elif reach == 2:
            expresions = [eqn.args[0] for eqn in self._equations]
            equations = [exp.subs(replace) for exp in expresions]
            jacobian = np.array(
                [
                    [sp.diff(eq, var) for eq in equations]
                    for var in self._variables
                ]
            )
            replace_values = [list(root) for root in roots]
            variable_list = list(self._variables.values())
            replace = [list(zip(variable_list, i)) for i in replace_values]
            a_matrices = []

            for j in replace:
                a_matrices.append(
                    np.array(
                        list(map(np.vectorize(lambda i: i.subs(j)), jacobian))
                    ).astype("float64")
                )

            w, v = np.linalg.eig(a_matrices)
            return w
        elif reach == 3:
            v = np.array([i.T for i in v])
            return v
        else:
            return roots, w, v

    def fixed_points(self, p):
        return self._linear_analysis(p, 1)[0]


class Autonomous(Symbolic):
    def eigenvalues(self, p):
        return self._linear_analysis(p, 2)

    def eigenvectors(self, p):
        return self._linear_analysis(p, 3)

    def full_linearize(self, p):
        return self._linear_analysis(p, 4)


"""class OneDim(Autonomous):


class TwoDim(Autonomous):


class ThreeDim(Autonomous):


class Nonautonomous(Symbolic):
"""


class AutoSymbolic(Symbolic):
    def __init__(self):
        cls = type(self)
        super().__init__(
            x=cls._variables,
            f=cls._functions,
            params=cls._parameters,
            name=cls._name,
        )


class Lorenz(AutoSymbolic):
    _name = "Lorenz"
    _variables = ["x", "y", "z"]
    _parameters = ["sigma", "rho", "beta"]
    _functions = ["sigma * (y - x)", "x * (rho - z) - y", "x * y - beta * z"]


class Duffing(AutoSymbolic):
    _name = "Duffing"
    _variables = ["x", "y", "t"]
    _parameters = ["alpha", "beta", "delta", "gamma", "omega"]
    _functions = [
        "y",
        "-delta * y - alpha * x - beta * x**3 + gamma * cos(omega * t)",
        "alpha"
    ]


class Logistic(AutoSymbolic):
    _name = "Logistic"
    _variables = ["N"]
    _parameters = ["r", "k"]
    _functions = ["r * N * (1 - N / k)"]
