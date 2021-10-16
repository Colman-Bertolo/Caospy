# from matplotlib import pyplot as plt
# from sympy import *

import types

import numpy as np
import pandas as pd
import sympy as sp
from scipy.integrate import solve_ivp
from sympy.parsing.sympy_parser import parse_expr


class Functional:
    def __init__(self, func, name):
        if not isinstance(func, types.FunctionType):
            raise Exception(
                "The first argument must be a callable"
                + f"got {type(func)} instead."
            )
        if not isinstance(name, str):
            raise Exception(
                "The second argument must be a string"
                + f"got {type(name)} instead."
            )
        self.func = func
        self.name = name

    def time_evolution(
        self,
        x0,
        parameters,
        ti=0,
        tf=200,
        rel_tol=1e-10,
        abs_tol=1e-12,
        mx_step=0.004,
    ):
        if not tf > ti:
            raise Exception(
                "Final integration time must be"
                + "greater than initial integration time."
            )
        sol = solve_ivp(
            self.func,
            [ti, tf],
            x0,
            args=(parameters,),
            rtol=rel_tol,
            atol=abs_tol,
            max_step=mx_step,
        )
        return sol.t, sol.y

    def poincare(
        self,
        x0,
        parameters,
        t_desc=5000,
        t_calc=50,
        rel_tol=1e-10,
        abs_tol=1e-12,
        mx_step=0.01,
    ):
        sol_1 = solve_ivp(
            self.func,
            [0, t_desc],
            x0,
            args=(parameters,),
            rtol=rel_tol,
            atol=abs_tol,
            max_step=mx_step,
        )
        x0_2 = [sol_1.y[i, -1] for i in range(np.shape(sol_1.y)[-1])]
        sol_2 = solve_ivp(
            self.func,
            [0, t_calc],
            x0_2,
            args=(parameters,),
            rtol=rel_tol,
            atol=abs_tol,
            max_step=mx_step,
        )
        variables = list(self._variables.values())
        return Poincare(sol_2.t, sol_2.y, variables)


class Symbolic(Functional):
    def __init__(self, x, f, params, name):
        if not isinstance(name, str):
            raise Exception(
                f"Name must be a string, got {type(name)} instead."
            )
        if not all(isinstance(i, list) for i in [x, f, params]):
            raise Exception(
                "The variables, functions and parameters"
                + "should be lists, got"
                + f"{(type(x), type(f), type(params))} instead."
            )
        for i in [x, f, params]:
            if not all(isinstance(j, str) for j in i):
                raise Exception("All the elements must be strings.")
        if not len(x) == len(f):
            raise Exception(
                "System must have equal number of variables"
                + f"and equations, insead has {len(x)} variables"
                + f"and {len(f)} equations"
            )

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
        for i in range(len(self._equations)):
            try:
                function.append(self._equations[i].args[0])
            except IndexError:
                function.append(parse_expr(self.f[i], local.copy(), {}))

        dydt = sp.lambdify(([*self._variables], [*self._parameters]), function)

        def fun(x_fun, t_fun, par_fun):
            return dydt(x_fun, par_fun)

        super().__init__(fun, self._name)

    def _linear_analysis(self, p, initial_guess, reach=3):
        if len(initial_guess) == 0:
            initial_guess = [0 for i in self._variables.values()]

        parameter_list = list(self._parameters.values())
        replace = list(zip(parameter_list, p))
        equalities = [eqn.subs(replace) for eqn in self._equations]
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
                raise Exception(
                    "Initial guess is not allowed,"
                    + "try with another set of values"
                )
        if len(roots) == 0:
            return [None, None, None, None]

        if isinstance(roots, dict):
            var_values = list(self._variables.values())
            roots_keys = list(roots.keys())
            if var_values != roots_keys:
                for k in var_values:
                    if k not in roots_keys:
                        roots[k] = k

            roots = [tuple(roots.values())]

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
        a_matrices = []
        for j in replace:
            try:
                a_matrices.append(
                    np.array(
                        list(map(np.vectorize(lambda i: i.subs(j)), jacobian))
                    ).astype("float64")
                )
            except TypeError:
                try:
                    a_matrices.append(
                        np.array(
                            list(
                                map(
                                    np.vectorize(lambda i: i.subs(j)), jacobian
                                )
                            )
                        ).astype("complex128")
                    )
                except TypeError:
                    a_matrices.append(
                        np.array(
                            list(
                                map(
                                    np.vectorize(lambda i: i.subs(j)), jacobian
                                )
                            ),
                            dtype=object,
                        )
                    )
        if reach == 5:
            a_matrices = np.array(a_matrices)
            return [None, a_matrices, None, None]

        elif reach == 2:
            w, v = np.linalg.eig(a_matrices)
            return [None, None, w, None]

        elif reach == 3:
            w, v = np.linalg.eig(a_matrices)
            v = np.array([i.T for i in v])
            return [None, None, v, None]
        else:
            w, v = np.linalg.eig(a_matrices)
            v = np.array([i.T for i in v])
            return [roots, a_matrices, w, v]

    def fixed_points(self, p, initial_guess=[]):
        return self._linear_analysis(p, initial_guess, 1)[0]


class MultiVarMixin(Symbolic):
    def eigenvalues(self, p, initial_guess=[]):
        return self._linear_analysis(p, initial_guess, 2)[2]

    def eigenvectors(self, p, initial_guess=[]):
        return self._linear_analysis(p, initial_guess, 3)[3]

    def full_linearize(self, p, initial_guess=[]):
        return self._linear_analysis(p, initial_guess, 4)


class OneDimMixin(Symbolic):
    def stability(self, parameters):
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
                raise Exception(
                    """Linear stability is undefined.
                    Slope in fixed point is 0."""
                )

            stable.append(True if slope < 0 else False)
        data = pd.DataFrame(
            list(zip(zero, slopes, stable)),
            columns=["Fixed Point", "Slope", "Stability"],
        )
        return data


class OneDim(OneDimMixin, Symbolic):
    def __init__(self, x, f, params, name):
        if not (len(x) == 1 & len(f) == 1):
            raise Exception(
                f"System shape is {len(x)} by {len(f)} but it should be 1 by 1"
            )
        super().__init__(x, f, params, name)


class TwoDimMixin(MultiVarMixin, Symbolic):
    def fixed_point_classify(self, params_values, initial_guess=[]):
        a = self._linear_analysis(params_values, initial_guess, reach=5)[1]
        if a is None:
            return "There is no fixed points to evaluate"

        traces = []
        dets = []
        classification = []
        for j in a:
            trace = j[0][0] + j[1][1]
            det = j[0][0] * j[1][1] - j[1][0] * j[0][1]
            traces.append(np.around(complex(trace), 2))
            dets.append(np.around(complex(det), 2))
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
                        + "Line of unstable fixed points"
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
                    if j[0][1] == 0 and j[1][0] == 0:
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


class TwoDim(TwoDimMixin, Symbolic):
    def __init__(self, x, f, params, name):
        if not (len(x) == 2 & len(f) == 2):
            raise Exception(
                f"System shape is {len(x)} by"
                + f"{len(f)} but it should be 2 by 2"
            )
        super().__init__(x, f, params, name)


class MultiDim(MultiVarMixin, Symbolic):
    def __init__(self, x, f, params, name):
        if not (len(x) == 3 & len(f) == 3):
            raise Exception(
                f"System shape is {len(x)} by"
                + f"{len(f)} but it should be 3 by 3"
            )
        super().__init__(x, f, params, name)


class AutoSymbolic(Symbolic):
    def __init__(self):
        cls = type(self)
        super().__init__(
            x=cls._variables,
            f=cls._functions,
            params=cls._parameters,
            name=cls._name,
        )


class Lorenz(MultiVarMixin, AutoSymbolic):
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
    ]


class Logistic(OneDimMixin, AutoSymbolic):
    _name = "Logistic"
    _variables = ["N"]
    _parameters = ["r", "k"]
    _functions = ["r * N * (1 - N / k)"]


class Trajectory:
    def __init__(self, t, x, variables):
        self.x = x  # Devuelve matriz de trayectorias x
        self.t = t  # Devuelve vector de tiempo t
        self.n_var = np.shape(x)[1]
        self.cols = ["t"] + [f"{v}" for v in variables]

    def to_table(self):
        col_names = self.cols
        pd.set_option("display.precision", 2)
        merge_array = np.insert(self.x, 0, self.t, axis=1)
        trajectory_table = pd.DataFrame(merge_array, columns=col_names)
        return trajectory_table


class Poincare(Trajectory):
    def __init__(self, t, x, variables):
        super().__init__(t, x, variables)

    def _fit(self, a, plane, grade, axis):
        if not plane >= 1 and plane < 4:
            raise Exception("Specified plane must be between 1 and 3")
        if not axis >= 1 and axis < 4:
            raise Exception("Specified axis must be between 1 and 3")

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
    def __init__(self, t, n, i, plane, axis):
        self.n0 = n[0]
        self.n1 = n[1]
        self.iterations = i
        self.t_iter = t
        self.plane = plane
        self.axis = axis
