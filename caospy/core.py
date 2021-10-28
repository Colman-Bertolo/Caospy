# from matplotlib import pyplot as plt
# from sympy import *


# Necessary packages and libraries are imported


import numpy as np
import pandas as pd
import sympy as sp
from scipy.integrate import solve_ivp
from sympy.parsing.sympy_parser import parse_expr
from .trajectories import *
from .poincare import *

# ==============================================================================

# Class Functional: this class can accept any system defined by its function and name. It is the higher class in the hierarchy of the core file.

class Functional:
    def __init__(self, func, name):
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

    # Method time_evolution: intergates numerically the function of the system

    def time_evolution(
        self,
        x0,
        parameters,
        ti=0,
        tf=200,
        n=2000,
        rel_tol=1e-10,
        abs_tol=1e-12,
        mx_step=0.004,
    ):
        if not tf > ti:
            raise ValueError(
                "Final integration time must be"
                + "greater than initial integration time."
            )
        t = np.linspace(ti, tf, n)
        sol = solve_ivp(
            self.func,
            [ti, tf],
            x0,
            args=(parameters,),
            rtol=rel_tol,
            atol=abs_tol,
            max_step=mx_step,
            dense_output=True,
            t_eval=t
        )
        x = sol.y
        variables = [f'$x_{i}$' for i in range(len(x0))]
        return Trajectory(t, x, variables)

    # Method poincare: integrates numerically a trajectory, eliminating the transient phase, to then instanciate a Poincare type object.

    def poincare(
        self,
        x0,
        parameters,
        t_desc=5000,
        t_calc=50,
        n=2000,
        rel_tol=1e-10,
        abs_tol=1e-12,
        mx_step=0.01,
    ):
        t1, x1 = self.time_evolution(
            x0,
            parameters,
            0,
            t_desc,
            n,
            rel_tol,
            abs_tol,
            mx_step
            )
        x0_2 = x1[:, -1]
        t2, x2 = self.time_evolution(
            x0_2,
            parameters,
            0,
            t_calc,
            n,
            rel_tol,
            abs_tol,
            mx_step
            )
        variables = list(self._variables.values())
        return Poincare(t2, x2, variables)

# ==========================================================================

# Class Symbolic: it defines a system by its equations, parameters, variables and name, given as strings. It inherits from Functional class.

class Symbolic(Functional):
    def __init__(self, x, f, params, name):
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

        def fun(t_fun, x_fun, par_fun):
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
                raise TypeError(
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
        n_var = len(self._variables)
        n_eq = len(self._equations)
        n_roots = len(replace)
        a_matrices = np.zeros((n_roots, n_eq, n_var), dtype=object)
        for i, rep in enumerate(replace):
            for j, row in enumerate(jacobian):
                for k, derivative in enumerate(row):
                    a_matrices[i, j, k] = derivative.subs(rep)

        try:
            a_matrices = a_matrices.astype('float64')
        except TypeError:
            try:
                a_matrices = a_matrices.astype('complex128')
            except TypeError:
                pass

        if reach == 5:
            return [None, a_matrices, None, None]

        elif reach == 2:
            w, v = np.linalg.eig(a_matrices)
            return [None, None, w, None]

        elif reach == 3:
            w, v = np.linalg.eig(a_matrices)
            v = next((i.T for i in v))
            return [None, None, None, v]
        else:
            w, v = np.linalg.eig(a_matrices)
            v = next((i.T for i in v))
            return [roots, a_matrices, w, v]

    def fixed_points(self, p, initial_guess=[]):
        return self._linear_analysis(p, initial_guess, 1)[0]

# ==========================================================================

# Class MultiVarMixin: this mixin takes the _linear_analysis method from Symbolic, and specifies it into three methods that are characteristic from multidimensional systems (eigenvalues, eigenvectors, full_linearize). It inherits from Symbolic.

class MultiVarMixin(Symbolic):
    def eigenvalues(self, p, initial_guess=[]):
        return self._linear_analysis(p, initial_guess, 2)[2]

    def eigenvectors(self, p, initial_guess=[]):
        return self._linear_analysis(p, initial_guess, 3)[3]

    def full_linearize(self, p, initial_guess=[]):
        return self._linear_analysis(p, initial_guess, 4)

# ==========================================================================

# Class OneDimMixin: this mixin incorporates two methods specific to onedimensional systems. It inherits from Symbolic.

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

# Class OneDim: takes a onedimensional system and gives it the specific functionalities that characterize them. It inherits from OneDimMixin and Symbolic.

class OneDim(OneDimMixin, Symbolic):
    def __init__(self, x, f, params, name):
        if not (len(x) == 1 & len(f) == 1):
            raise Exception(
                f"System shape is {len(x)} by {len(f)} but it should be 1 by 1"
            )
        super().__init__(x, f, params, name)

# ==========================================================================

# Class TwoDimMixin: incorporates the classification of fixed points for two dimensional systems. It inherits from MultiVarMixin and Symbolic.

class TwoDimMixin(MultiVarMixin, Symbolic):
    def fixed_point_classify(self, params_values, initial_guess=[]):
        a = self._linear_analysis(params_values, initial_guess, reach=5)[1]
        roots = self.fixed_points(params_values, initial_guess)
        if a is None:
            return "There is no fixed points to evaluate"

        traces = []
        dets = []
        classification = []
        for i, r in enumerate(roots):
            if len(r) == 1:
                trace = a[0][0] + a[1][1]
                det = a[0][0] * a[1][1] - a[1][0] * a[0][1]
            else:
                trace = a[i][0][0] + a[i][1][1]
                det = a[i][0][0] * a[i][1][1] - a[i][1][0] * a[i][0][1]

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

# Class TwoDim: takes a twodimensional system and gives it the specific functionalities that characterize them. It inherits from TwoDimMixin and Symbolic.

class TwoDim(TwoDimMixin, Symbolic):
    def __init__(self, x, f, params, name):
        if not (len(x) == 2 & len(f) == 2):
            raise ValueError(
                f"System shape is {len(x)} by"
                + f"{len(f)} but it should be 2 by 2"
            )
        super().__init__(x, f, params, name)

# ==========================================================================

# Class MultiDim: takes a multidimensional system and gives it the specific functionalities that characterize them. It inherits from MultiVarMixin and Symbolic.

class MultiDim(MultiVarMixin, Symbolic):
    def __init__(self, x, f, params, name):
        if not (len(x) == 3 & len(f) == 3):
            raise ValueError(
                f"System shape is {len(x)} by"
                + f"{len(f)} but it should be 3 by 3"
            )
        super().__init__(x, f, params, name)

# ==========================================================================

# Class AutoSymbolic: it takes predetermined classes that define particular systems and instanciates as Symbolic objects. It inherits from Symbolic.

class AutoSymbolic(Symbolic):
    def __init__(self):
        cls = type(self)
        super().__init__(
            x=cls._variables,
            f=cls._functions,
            params=cls._parameters,
            name=cls._name,
        )

class LinearityError(ValueError):
    pass
