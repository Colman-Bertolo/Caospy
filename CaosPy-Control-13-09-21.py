import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sympy as sp
import pandas as pd

'''Class - SymbolicSystem : It takes any dynamical system defined by its variables, equations, parameters and name.
The initialization requires three lists of strings and an isolated string, the first with the variables names, second
with the functions and third with parameters names. Finally one gives the system's name as a string.

Attributes :
- self.x: returns variable names list.
- self.params : returns parameter names list.
- self.name : returns system's name.
- self.f : returns functions list.
- self.n_var : returns system's number of variables
- self.n_eq : returns system's number of equations.
- self.system_shape : returns string indicating both quantities mentioned above.


Methods:

- __init__ : takes three lists of strings and one isolated string. First, a list with the variables names, second,
a list with the equations of the system, third, a list with the parameters names and finally, the system's name.

- fixed_points: takes a set of parameters corresponding to the system and returns array of the fixed points for
the given values.

- eigen: takes a set of parameters corresponding to the system and returns a list filled with tuples, which
first element is the eigenvalue and the second are the coordinates of the eigenvector.

'''


class SymbolicSystem:
    def __init__(self, x, f, params, name):
        self.x = x
        self.params = params
        self.name = name.title()
        self.f = f
        self.n_var = len(x)
        self.n_eq = len(f)
        self.system_shape = f'The system has {len(f)} equations and {len(x)} variables'

    def fixed_points(self, p):
        v_str = ', '.join(self.x)  # Construct single string with variable names.
        v = sp.symbols(v_str)  # Uses sympy library for declaring symbolic variables of the system.
        locals().update(dict(zip(self.x, v)))  # Declare local variables, corresponding to system's variables names.
        locals().update(dict(zip(self.params, p)))  # Same as above, but with parameters, and instead of assigning them
        # symbolic values, it gives them the specified "p" parameters.

        func = []  # Constructs a list with the functions, using eval() method to transform them from string to
        # executable code.
        for derivative in self.f:
            func.append(sp.Eq(eval(derivative), 0))

        roots = sp.solve(func, v)  # Uses numpy solve method to get the fixed points.
        if isinstance(roots, list):
            roots = np.array(roots)  # If roots came in form of a list, then they are turned into an array.

        elif isinstance(roots, dict):
            roots = np.array([i for i in roots.values()])  # If roots came in form of a dictionary, they are
            # transformed into a numpy array.

        return roots

    def eigen(self, p):
        v_str = ', '.join(self.x)
        v = sp.symbols(v_str)
        locals().update(dict(zip(self.x, v)))
        locals().update(dict(zip(self.params, p)))

        equations = []
        for func in self.f:
            equations.append(eval(func))  # Up to here, the process is the same that for the root solving.
        jacobian = np.array([[sp.diff(eq, var) for eq in equations] for var in v])  # The Jacobian is declared, confor-
        # med by a numpy array of the computed derivatives.
        points = [list(zip(v, i)) for i in self.fixed_points(p)]  # The list of fixed points, given in tuples of the
        # variable and the respective value of it, for that given fixed point.
        a_matrices = []
        for J in points:
            a_matrices.append(np.array(list(map(np.vectorize(lambda i: i.subs(J)), jacobian))))  # Appending the
        # evaluated derivative in the respective fixed point into a list.

        a_matrices = np.array(list(map(np.vectorize(lambda i: float(i)), a_matrices)))  # Converting list into array.

        w, v = np.linalg.eig(a_matrices)  # Applying linalg.eig from numpy to get two lists, one with the eigenvalues
        # and the other one with the eigenvectors.

        eigen_both = [(w[i][j], v[i][:, j]) for i in range(np.shape(a_matrices)[0]) for j in
                      range(np.shape(a_matrices)[-1])]

        # Finally returning a list of tuples.
        return eigen_both


''' Class - DynamicSystem: it gives instance to a dynamical system, defined by a function containing its equations, and
a set of parameters, along with the system's name.

Attributes:
- f : returns system's function.
- c : returns system's set of defined parameters as a list.
- name : returns string with system's name
- c_tuple : returns tuple of parameters.

Methods:
- __init__ : it takes a function 'f', a list of parameters 'c' and the system's name.

 -  time_evolution: it takes a list of initial conditions x0, an integration step size, and a number of steps to cal -
 culate, and returns a Trajectory class object. Defined by its time vector 't', and state matrix 'x'.

'''


class DynamicSystem:
    def __init__(self, f, c, name):
        self.f = f
        self.c = c
        self.c_tuple = tuple(c)
        self.name = name

    def time_evolution(self, x0, step=0.01, n=5000):
        t = np.linspace(0, step * n, n)
        integration = odeint(self.f, x0, t, args=self.c_tuple)
        trajectory = Trajectory(t, integration)
        return trajectory


'''Class Lorenz: it's a subclass from DynamicSystem, but specialized to the Lorenz systems. It has its own classmethod
function, named func_lorenz, called when integration needed. And it is initialized with the values of the three charac-
teristic parameters, sigma, rho and beta.

Attributes:
- sigma: returns value of sigma.
- rho: // // rho.
- beta: // // beta.

Methods:
- __init__: it takes sigma, rho and beta in that order.

- func_lorenz (@classmethod): it takes a state v, a set of args, p and returns the derivatives of x, y and z.

- poincare: it computes a trajectory and erase the transient, returning a Poincaré class object.


'''


class Lorenz(DynamicSystem):
    def __init__(self, sigma, rho, beta):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        super().__init__(self.func_lorenz, [sigma, rho, beta], name='Lorenz')  # Initialize superclass, in the given
        # format.

    @classmethod
    def func_lorenz(cls, v, t, *p):
        x, y, z = v
        sigma, rho, beta = p
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return [dx, dy, dz]

    def poincare(self, x0, t_desc=20000, t_calc=50, step=0.01):
        t_discard = np.linspace(0, t_desc, int(t_desc / step))
        parameters_tuple = (self.sigma, self.rho, self.beta)
        x = odeint(self.func_lorenz, x0, t_discard, args=parameters_tuple)  # Integration of discarded transient.
        t_calculate = np.linspace(0, t_calc, int(t_calc / step))
        x0 = x[-1]
        x = odeint(self.func_lorenz, x0, t_calculate, args=parameters_tuple)  # Integration of used stationary values.
        p = Poincare(t_calculate, x)  # Initializing Poincaré class element.
        return p


''' Class Duffing: Like Lorenz, is a subclass of DynamicSystem, it takes the five Duffing parameters and represents a
system of this nature.

Attributes:
- alpha: returns alpha parameter.
- beta: returns beta parameter.
- delta: returns delta parameter.
- gamma: returns gamma parameter.
- omega: returns omega parameter.

Methods:
- func_duffing (@classmethod): takes a state v, a set of arguments p and computes the derivatives of x and y.

'''


class Duffing(DynamicSystem):
    def __init__(self, alpha, beta, delta, gamma, omega):
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.gamma = gamma
        self.omega = omega
        super().__init__(self.func_duffing, [alpha, beta, delta, gamma, omega], name='Duffing')

    @classmethod
    def func_duffing(cls, v, t, *p):
        x, y = v
        alpha, beta, delta, gamma, omega = p
        dx = y
        dy = -delta * y - alpha * x - beta * x ** 3 + gamma * np.cos(omega * t)
        return [dx, dy]


''' Class Trajectory: It is defined by a time vector 't' and a state matrix 'x'. From them we can work on the
trajectories integrated and analyze them.

Attributes:
- x: returns state matrix x.
- t: returns time vector t.
- n_var: returns number of variables of the system.
- cols: returns the name of the columns as a list of strings.

Methods:

- __init__ : takes a vector of time and a matrix of states.

- to_table: it receives a list of column names as strings, and gives back a pandas DataFrame with the columns being 't'
and the system's variables, and rows being every integration step along the trajectory.

- plot_trajectory3d: it takes a plot size, and gives back the 3D plot of the system (if it is three dimensional).

- plot_trajectory2d: it takes a tuple of indices, corresponding to the number of the axis of each variable, and a
size figure given as a tuple also. And plots the two specified variables in the axes.

- plot_x1t: takes a figsize as a tuple and returns plot of first variable of the system against time.

- plot_x2t: takes a figsize as a tuple and returns plot of second variable of the system against time.

- plot_x2t: takes a figsize as a tuple and returns plot of third variable of the system against time.

'''


class Trajectory:
    def __init__(self, t, x):
        self.x = x  # Devuelve matriz de trayectorias x
        self.t = t  # Devuelve vector de tiempo t
        self.n_var = np.shape(x)[1]
        self.cols = ['t'] + [f'x{i + 1}' for i in range(self.n_var)]

    def to_table(self, col_names):
        col_names = ['t'] + [f'x{i + 1}' for i in range(self.n_var)] if col_names is None else ['t'] + col_names
        trajectory_array = np.hstack((np.array([self.t]).T, self.x))
        trajectory_table = pd.DataFrame(trajectory_array, columns=col_names)
        return trajectory_table

    def plot_trajectory3d(self, size=(5, 5)):
        assert self.n_var == 3, f'Number of variables must be 3, instead got {self.n_var}'
        fig = plt.figure(figsize=size)
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(self.x[:, 0],
                self.x[:, 1],
                self.x[:, 2])
        ax.set_title('Lorenz Attractor')
        ax.set_xlabel('$x_{1}$')
        ax.set_ylabel('$x_{2}$')
        ax.set_zlabel('$x_{3}$')

    def plot_trajectory2d(self, variables=(0, 1), size=(5, 5)):
        assert self.n_var >= 2, f'Number of variables must be greater or equal to 2, instead got {self.n_var}'
        fig, ax = plt.subplots(figsize=size)
        ax.plot(self.x[:, variables[0]], self.x[:, variables[1]], 'b-')
        ax.set_title(f'$x_{variables[0] + 1} - x_{variables[1] + 1}$')
        ax.set_ylabel(f'$x_{variables[1] + 1}$')
        ax.set_xlabel(f'$x_{variables[0] + 1}$')

    def plot_x1t(self, size=(5, 5)):
        fig, ax = plt.subplots(figsize=size)
        ax.plot(self.t, self.x[:, 0], 'k-', label='$x_{1}(t)$')
        ax.legend()
        ax.set_title('$x_{1} - t$')
        ax.set_xlabel('$t$')
        ax.set_ylabel('$x_{1}(t)$')

    def plot_x2t(self, size=(5, 5)):
        assert self.n_var >= 2, f'Number of variables must be greater or equal to 2, instead got {self.n_var}'
        fig, ax = plt.subplots(figsize=size)
        ax.plot(self.t, self.x[:, 1], 'k-', label='$x_{2}(t)$')
        ax.legend()
        ax.set_title('$x_{2} - t$')
        ax.set_xlabel('$t$')
        ax.set_ylabel('$x_{2}(t)$')

    def plot_x3t(self, size=(5, 5)):
        assert self.n_var == 3, f'Number of variables must be 3, instead got {self.n_var}'
        fig, ax = plt.subplots(figsize=size)
        ax.plot(self.t, self.x[:, 2], 'k-', label='$x_{3}(t)$')
        ax.legend()
        ax.set_title('$x_{3} - t$')
        ax.set_xlabel('$t$')
        ax.set_ylabel('$x_{3}(t)$')


'''Class Poincare: it represents trajectories of systems susceptible of being analyzed through the poincare map method.
Takes a vector of time and a matrix of states, and computes the poincaré map, either through a model fitting of first
or second order, or through the Henon method, yet to be added to the code.

Attributes:
- t: returns time vector t.
- x: returns state matrix x.

Methods:
- _fit ('Private'): Internal method of the class, used to compute the poincaré maps of the different trajectories.
Takes a constant value 'a', which will be the value at which the map will be evaluated. It takes a plane value, that
indicates to which of the variable axis the plane will be perpendicular to. A grade value, indicating the model fitting
shape, and an axis value, to indicate which variable will be iterated. Returns a time vector of the map t_map,
a matrix of the nth and nth + 1 state of the system (the map) and a vector of the iterations of the leaving variable.

- z_map: it returns a map object of the z variable, initialized with the return of the _fit method, and a plane and
axis values. The method supports changing the a parameter, the axis of the third variable and the grade of the
polynomial fitting.

- x_map: same as z_map but returns x map.

- y_map: same as z_map but returns y map.
'''


class Poincare:
    def __init__(self, t, x):
        self.t = t
        self.x = x

    def _fit(self, a, plane, grade, axis):
        assert 1 <= plane < 4  # Ensuring that plane and axis don't fall outside the allowed interval.
        assert 1 <= axis < 4

        # Erasing last two elements of time vector two match length of state slices matrices.
        t = np.delete(self.t, [-1, -2])

        # Concatenating three consecutive states in array of columns xi, xi+1 and xi+2. Variable corresponds to axis.
        x1a = np.delete(np.roll(self.x[:, axis - 1], 2), [0, 1])
        x1b = np.delete(np.roll(self.x[:, axis - 1], 1), [0, 1])
        x1c = np.delete(self.x[:, axis - 1], [0, 1])

        x1_slices = np.vstack((x1a, x1b, x1c))

        # Concatenating three consecutive states in array of columns xi, xi+1 and xi+2. Variable corresponds to plane.
        x2a = np.delete(np.roll(self.x[:, plane - 1], 2), [0, 1])
        x2b = np.delete(np.roll(self.x[:, plane - 1], 1), [0, 1])
        x2c = np.delete(self.x[:, plane - 1], [0, 1])

        x2_slices = np.vstack((x2a, x2b, x2c))

        # Keeping only columns that lie in the 'a' plane crossing.
        x1 = x1_slices[:, (x1_slices[0] < a) & (x1_slices[1] > a)]
        x2 = x2_slices[:, (x1_slices[0] < a) & (x1_slices[1] > a)]
        t_map = t[(x1_slices[0] < a) & (x1_slices[1] > a)]

        # Stacking together x_axis and x_plane consecutive points.
        x12 = np.vstack((x2, x1))

        # Getting fitting coefficients.
        def poly(v, grade_poly):
            return np.polyfit(v[3:6], v[0:3], grade_poly)

        x12_coefficients = np.apply_along_axis(poly, 0, x12, grade)

        # Fitting points of trajectory.
        def apply_poly(p, a_value):
            return p[0] * a_value ** 2 + p[1] * a_value + p[2] if len(p) == 3 else p[0] * a_value + p[1]

        x2_fit = np.apply_along_axis(apply_poly, 0, x12_coefficients, a)

        x21 = np.delete(x2_fit, -1)
        x22 = np.delete(x2_fit, 0)

        # Creating array containing the map.
        x2_map = np.array([[x21], [x22]])

        return t_map, x2_map, x2_fit

    def z_map(self, a=0, axis=1, grade=1):
        plane = 3
        t_map, xi_map, x2_fit = self._fit(a, plane, grade, axis)
        map_z = Map(t_map, xi_map, x2_fit, plane, axis)
        return map_z

    def y_map(self, a=0, axis=1, grade=1):
        plane = 2
        t_map, xi_map, x2_fit = self._fit(a, plane, grade, axis)
        map_y = Map(t_map, xi_map, x2_fit, plane, axis)
        return map_y

    def x_map(self, a=0, axis=2, grade=1):
        plane = 1
        t_map, xi_map, x2_fit = self._fit(a, plane, grade, axis)
        map_x = Map(t_map, xi_map, x2_fit, plane, axis)
        return map_x


'''Class Map: it represents a poincaré map, and can be treated as one.

Attributes:
- n0: returns array with Nth state of the variable.
- n1: returns array with Nth + 1 state of the variable.
- iterations: returns array of fitted points of the axis variable.
- t_iter: returns time vector of iterations.
- plane: returns number of the plane corresponding to the variable of the map.
- axis: returns number of the axis corresponding to the iterated variable.
- n = returns full map array.

Methods:
- plot_cobweb: it gives back the cobweb diagram of the computed map.

- plot_iterations: it returns the plot of the axis variable against the number of iterations.
'''


class Map:
    def __init__(self, t, n, i, plane, axis):
        self.n0 = n[0][0]
        self.n1 = n[1][0]
        self.iterations = i
        self.t_iter = t
        self.plane = plane
        self.axis = axis
        self.n = n

    def plot_cobweb(self):
        title = 'Z map' if self.plane == 3 else 'Y map' if self.plane == 2 else 'X map'
        x_label = 'z(i)' if self.plane == 3 else 'y(i)' if self.plane == 2 else 'x(i)'
        y_label = 'z(i+1)' if self.plane == 3 else 'y(i+1)' if self.plane == 2 else 'x(i+1)'
        fig, ax = plt.subplots(1, figsize=(7, 7))
        ax.plot(self.n0, self.n1, 'k.')
        ax.plot(np.linspace(0, 800, 800), np.linspace(0, 800, 800), 'k-')
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xlim(min(self.n0) - min(self.n0) * 0.05, max(self.n0) + max(self.n0) * 0.05)
        ax.set_ylim(min(self.n1) - min(self.n1) * 0.05, max(self.n1) + max(self.n1) * 0.05)

    def plot_iterations(self):
        title = 'Z(t)' if self.plane == 3 else 'Y(t)' if self.plane == 2 else 'X(t)'
        y_label = 'y(i)' if self.plane == 3 and self.axis == 1 or self.plane == 1 and self.axis == 3 else 'x(i)' if \
            self.plane == 3 and self.axis == 2 or self.plane == 2 and self.axis == 3 else 'z(i)' if self.plane == 2 \
            and self.axis == 1 or self.plane == 1 and self.axis == 2 else None
        fig, ax = plt.subplots(1, figsize=(7, 7))
        ax.plot(self.t_iter, self.iterations, 'k.')
        ax.set_title(title)
        ax.set_xlabel('n')
        ax.set_ylabel(y_label)
