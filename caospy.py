import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sympy as sp
import pandas as pd


class Symbolic:
    def __init__(self, x, f, params, name):
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
                eq = sp.Eq(eval(eq, {}, local.copy()), 0)
            self._equations.append(eq)


    def fixed_points(self, p):
        parameter_list = list(self._parameters.values())
        replace = list(zip(parameter_list, p))
        equations = [eqn.subs(replace) for eqn in self._equations]
        roots = sp.solve(equations, self._variables)

        if isinstance(roots, list):
            roots = np.array(roots)

        elif isinstance(roots, dict):
            roots = np.array([i for i in roots.values()])

        return roots

    def eigenvalues(self, p):
        parameter_list = list(self._parameters.values())
        replace = list(zip(parameter_list, p))
        expresions = [eqn.args[0] for eqn in self._equations]
        equations = [exp.subs(replace) for exp in expresions]
        jacobian = np.array([[sp.diff(eq, var) for eq in equations] for var in self._variables])
        replace_values = [list(root.values()) for root in self.fixed_points(p)]
        variable_list = list(self._variables.values())
        replace = [list(zip(variable_list, i)) for i in replace_values]
        a_matrices = []

        for j in replace:
            a_matrices.append(np.array(
                list(
                    map(np.vectorize(lambda i: i.subs(j)), jacobian)
                     )
                        ).astype('float64'))

        w, v = np.linalg.eig(a_matrices[0])
        #eigen_both = [(w[i][j], v[i][:, j]) for i in range(np.shape(a_matrices)[0]) for j in
                      #range(np.shape(a_matrices)[-1])]

        return w
