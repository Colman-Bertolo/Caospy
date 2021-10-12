# from sympy import *
# import matplotlib.pyplot as plt

import types

import numpy as np
import pandas as pd
import sympy as sp
from scipy.integrate import odeint
from sympy.parsing.sympy_parser import parse_expr


class Functional:
    def __init__(self, func, name):
        assert isinstance(
            func, types.FunctionType
        ), "The first argument must be of the type 'Function',\n"
        f"got {type(func)} instead."
        assert isinstance(
            name, str
        ), "The second argument must be of the type 'string',\n"
        f"got {type(name)} instead."

        self.func = func
        self.name = name

    def time_evolution(self, x0, parameters, ti=0, tf=200, n=2000):
        assert (
            tf > ti
        ), """Final integration time must be greater than
        initial integration time."""
        t = np.linspace(ti, tf, int(n))
        y = odeint(self.func, x0, t, args=(parameters,))
        return [t, y]

    def poincare(self, x0, parameters, t_desc=5000, t_calc=50, step=0.01):
        t_discard = np.linspace(0, t_desc, int(t_desc / step))
        x = odeint(self.func, x0, t_discard, args=(parameters,))
        t_calculate = np.linspace(0, t_calc, int(t_calc / step))
        x0 = [x[i, -1] for i in range(np.shape(x)[-1])]
        x = odeint(self.func, x0, t_calculate, args=(parameters,))
        variables = list(self._variables.values())
        return Poincare(t_calculate, x, variables)


class Symbolic(Functional):
    def __init__(self, x, f, params, name):
        assert isinstance(
            name, str
        ), f"Name must be a string, got {type(name)} instead."
        assert all(
            isinstance(i, list) for i in [x, f, params]
        ), "The variables, functions and parameters"
        "should be lists, got"
        f"{(type(x), type(f), type(params))} instead."
        for i in [x, f, params]:
            assert all(
                isinstance(j, str) for j in i
            ), "All the elements must be strings."
        assert len(x) == len(
            f
        ), f"""System must have equal number of variables
        and equations, insead has {len(x)} variables
        and {len(f)} equations"""

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
        roots = sp.solve(equalities, list(self._variables.values()))
        if isinstance(roots, dict):
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
            return roots

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
            return a_matrices

        elif reach == 2:
            w, v = np.linalg.eig(a_matrices)
            return w

        elif reach == 3:
            w, v = np.linalg.eig(a_matrices)
            v = np.array([i.T for i in v])
            return v
        else:
            w, v = np.linalg.eig(a_matrices)
            v = np.array([i.T for i in v])
            return roots, w, v

    def fixed_points(self, p):
        return self._linear_analysis(p, 1)


class MultiVarMixin(Symbolic):
    def eigenvalues(self, p):
        return self._linear_analysis(p, 2)

    def eigenvectors(self, p):
        return self._linear_analysis(p, 3)

    def full_linearize(self, p):
        return self._linear_analysis(p, 4)


class OneDimMixin(Symbolic):

    """def diff_plot(self, parameters, x_lim, dx_lim, n=1000):
    x_points = np.linspace(x_lim[0], x_lim[1], n)
    dx_points = []
    for x in x_points:
        dx_points.append(self.func((x, ), 1, parameters))

    plt.plot(x_points, dx_points)
    plt.title("Differential plot")
    plt.ylabel("d" + list(self._variables.keys())[0])
    plt.xlabel(list(self._variables.keys())[0])
    plt.xlim(x_lim)
    plt.ylim(dx_lim)
    plt.grid()
    plt.show()"""

    def stability(self, parameters):
        replace_params = list(zip(self._parameters.values(), parameters))
        equation = self._equations[0].args[0].subs(replace_params)
        derivative = sp.diff(equation, list(self._variables.values())[0])
        zero = self.fixed_points(parameters)
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
        assert (
            len(x) == 1 & len(f) == 1
        ), f"""System shape is {len(x)} by {len(f)} but it should be 1 by 1"""
        super().__init__(x, f, params, name)


class TwoDimMixin(MultiVarMixin, Symbolic):
    def fixed_point_classify(self, params_values):
        a = self._linear_analysis(params_values, reach=5)
        traces = []
        dets = []
        classification = []
        for j in a:
            trace = j[0][0] + j[1][1]
            det = j[0][0] * j[1][1] - j[1][0] * j[0][1]
            traces.append(np.around(complex(trace), 2))
            dets.append(np.around(complex(det), 2))
            if det == 0:
                classification.append("Non Isolated Fixed-Point")

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

        pd.set_option("display.precision", 2)
        roots = self.fixed_points(params_values)
        eigen = self.eigenvalues(params_values)
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

        data = pd.DataFrame(data_array, columns=cols)
        return data


class TwoDim(TwoDimMixin, Symbolic):
    def __init__(self, x, f, params, name):
        assert (
            len(x) == 2 & len(f) == 2
        ), f"""System shape is {len(x)} by {len(f)} but it should be 2 by 2"""
        super().__init__(x, f, params, name)


class ThreeDim(MultiVarMixin, Symbolic):
    def __init__(self, x, f, params, name):
        assert (
            len(x) == 3 & len(f) == 3
        ), f"""System shape is {len(x)} by {len(f)} but it should be 3 by 3"""
        super().__init__(x, f, params, name)


# class Nonautonomous(Symbolic):


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

    """def plot_trajectory3d(self, size=(5, 5)):
        assert (
            self.n_var == 3
        ), f"Number of variables must be 3, instead got {self.n_var}"
        fig = plt.figure(figsize=size)
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(self.x[:, 0], self.x[:, 1], self.x[:, 2])
        ax.set_title("Lorenz Attractor")
        ax.set_xlabel("$x_{1}$")
        ax.set_ylabel("$x_{2}$")
        ax.set_zlabel("$x_{3}$")

    def plot_trajectory2d(self, variables=(0, 1), size=(5, 5)):
        assert (
            self.n_var >= 2
        ), "Number of variables must be greater or equal to 2"
        f", instead got {self.n_var}"
        fig, ax = plt.figure(figsize=size)
        ax.plot(self.x[:, variables[0]], self.x[:, variables[1]], "k-")
        ax.set_title(f"$x_{variables[0] + 1}$ - x_{variables[1]}")
        ax.set_ylabel(f"$x_{variables[1]}$")
        ax.set_xlabel(f"$x_{variables[0]}$")

    def plot_x1t(self, size=(5, 5)):
        fig, ax = plt.subplots(figsize=size)
        ax.plot(self.t, self.x[:, 0], "$x_{1}$", label="$x_{1}(t)$")
        ax.legend()
        ax.set_title("$x_{1} - t$")
        ax.set_xlabel("$t$")
        ax.set_ylabel("$x_{1}(t)$")

    def plot_x2t(self, size=(5, 5)):
        assert (
            self.n_var >= 2
        ), "Number of variables must be greater or equal to 2,"
        f"instead got {self.n_var}"
        fig, ax = plt.subplots(figsize=size)
        ax.plot(self.t, self.x[:, 1], "$x_{2}$", label="$x_{2}(t)$")
        ax.legend()
        ax.set_title("$x_{2} - t$")
        ax.set_xlabel("$t$")
        ax.set_ylabel("$x_{2}(t)$")

    def plot_x3t(self, size=(5, 5)):
        assert (
            self.n_var == 3
        ), f"Number of variables must be 3, instead got {self.n_var}"
        fig, ax = plt.subplots(figsize=size)
        ax.plot(self.t, self.x[:, 2], "$x_{3}$", label="$x_{3}(t)$")
        ax.legend()
        ax.set_title("$x_{3} - t$")
        ax.set_xlabel("$t$")
        ax.set_ylabel("$x_{3}(t)$")"""


class Poincare(Trajectory):
    def __init__(self, t, x, variables):
        super().__init__(t, x, variables)

    def _fit(self, a, plane, grade, axis):
        assert plane >= 1 and plane < 4
        assert axis >= 1 and axis < 4

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

    def z_map(self, a=0, axis=1, grade=1):
        plane = 3
        fit = self._fit(a, plane, grade, axis)
        map_z = Map(fit, plane, axis)
        return map_z

    def y_map(self, a=0, axis=1, grade=1):
        plane = 2
        fit = self._fit(a, plane, grade, axis)
        map_y = Map(fit, plane, axis)
        return map_y

    def x_map(self, a=0, axis=2, grade=1):
        plane = 1
        fit = self._fit(a, plane, grade, axis)
        map_x = Map(fit, plane, axis)
        return map_x


class Map:
    def __init__(self, t, n, i, plane, axis):
        self.n0 = n[0]
        self.n1 = n[1]
        self.iterations = i
        self.t_iter = t
        self.plane = plane
        self.axis = axis

    """def plot_cobweb(self):
        title = (
            "Z map"
            if self.plane == 3
            else "Y map"
            if self.plane == 2
            else "X map"
        )
        xlabel = (
            "z(i)"
            if self.plane == 3
            else "y(i)"
            if self.plane == 2
            else "x(i)"
        )
        ylabel = (
            "z(i+1)"
            if self.plane == 3
            else "y(i+1)"
            if self.plane == 2
            else "x(i+1)"
        )
        fig, ax = plt.subplots(1, figsize=(7, 7))
        ax.plot(self.n0, self.n1, "r;")
        ax.plot(np.linspace(0, 800, 800), np.linspace(0, 800, 800), "k-")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    def plot_iterations(self):
        title = (
            "Z(t)"
            if self.plane == 3
            else "Y(t)"
            if self.plane == 2
            else "X(t)"
        )
        ylabel = (
            "y(i)"
            if self.plane == 3
            and self.axis == 1
            or self.plane == 1
            and self.axis == 3
            else "x(i)"
            if self.plane == 3
            and self.axis == 2
            or self.plane == 2
            and self.axis == 3
            else "z(i)"
            if self.plane == 2
            and self.axis == 1
            or self.plane == 1
            and self.axis == 2
            else None
        )
        fig, ax = plt.subplots(1, figsize=(7, 7))
        ax.plot(self.t_iter, self.iterations, "k.")
        ax.set_title(title)
        ax.set_xlabel("n")
        ax.set_ylabel(ylabel)"""
