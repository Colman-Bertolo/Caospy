import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sympy as sp
from scipy.optimize import fsolve
import pandas as pd


''' Clase 'NonlinearSystem': Representa un sistema cualquiera de los sistemas cargados en la librería. Se inicializa con vector condiciones iniciales 'x0', vector parámetros 'c' y nombre de sistema dentro de una lista de posibles sistemas (Lorenz, duffing, etc.)
'''


class SymbolicSystem:
    def __init__(self, x, f, params, name):
        self.x = x # Devuelve lista de variables
        self.params = params # Devuelve lista de parámetros
        self.name = name.title() # Devuelve cadena de caracteres con nombre
        self.f = f # Devuelve lista con las funciones del sistema
        self.n_var = len(x)
        self.n_eq = len(f)
        self.syst_shape = f'El sistema es de {len(f)} ecuaciones por {len(x)} incógnitas'


    def fixed_points(self, *p):
        v_str = ', '.join(self.x)
        v = sp.symbols(v_str)
        locals().update(dict(zip(self.x, v)))
        locals().update(dict(zip(self.params, p)))

        func = []
        for derivative in self.f:
           func.append(sp.Eq(eval(derivative), 0))

        roots = sp.solve(func, v)
        if isinstance(roots, list):
            roots = np.array(roots)

        elif isinstance(roots, dict):
            roots = np.array([i for i in roots.values()])

        return roots


    def eigenvalues(self):
        v_str = ', '.join(self.x)
        v = sp.symbols(v_str)
        locals().update(dict(zip(self.x, v)))
        locals().update(dict(zip(self.params, )))

        equations = [eval(func) for func in self.f]
        Jacobian = np.array([[sp.diff(eq, var) for eq in equations] for var in v])
        points = [list(zip(v, i)) for i in self.fixed_points]
        A_Matrices = []
        for J in points:
            A_Matrices.append(np.array(list(map(np.vectorize(lambda i: i.subs(J)), Jacobian))))

        return # Aún no sé como retornar

    #def eigenvectors(self):


'''def duffing(t, x, *c):
    x, y  = x
    alpha, beta, delta, gamma, omega = c
    dx = y
    dy = -delta * y - alpha * x - beta * x**3 + gamma * np.cos(omega * t)
    return [dx, dy]'''



'''
Clase Lorenz: Permite representar un sistema de Lorenz de 3x3. Se inicializa con condiciones iniciales x0, y0, z0 y parámetros sigma, rho y beta.'''

class DynamicSystem:
    def __init__(self, f, c, name):
        self.f = f
        self.c = c
        self.c_tuple = tuple(c)
        self.name = name

    def TimeEvolution(self, x0, step=0.01, n=5000):
        t = np.linspace(0, step * n, n)
        integration = odeint(self.f, self.x0, t, args=self.c_tuple)
        trajectory = Trajectory(t, integration)
        return trajectory

class Lorenz(DynamicSystem):
    def __init__(self, sigma, rho, beta):
        self.sigma = sigma # Devuelve parámetro sigma
        self.rho = rho # Devuelve parámetro rho
        self.beta = beta # Devuelve parámetro beta
        super().__init__(self.func_lorenz, [sigma, rho, beta], name='Lorenz')

    @classmethod
    def func_lorenz(cls, x, t, *p):
        x, y, z = x
        sigma, rho, beta = p
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return [dx, dy, dz]

    def poincare(self, t_desc=20000, t_calc=50, step=0.01):
        t_descarte = np.linspace(0, t_desc, int(t_desc / step))
        x = odeint(lorenz, self.X0, t_desc, self.params_tuple)
        t_calculo = np.linspace(0, t_calc, int(t_calc / step))
        x0 = [x[0, -1], x[1, -1], x[2, -1]]
        x = odeint(lorenz, x0, t_calculo, self.params_tuple)
        p = Poincare(t_calculo, x)
        return p


'''
Clase Duffing: Permite representar sistema de Duffing. Se inicializa a partir de condiciones iniciales x0 e y0 y parámetros alpha, beta, delta, gamma y omega.
'''


class Duffing(DynamicSystem):
    def __init__(self, alpha, beta, delta, gamma, omega):
        self.alpha = alpha # Devuelve parámetro alpha
        self.beta = beta # Devuelve parámetro beta
        self.delta = delta # Devuelve parámetro delta
        self.gamma = gamma # Devuelve parámetro gamma
        self.omega = omega # Devuelve parámetro omega
        super().__init__(self.func_duffing, [alpha, beta, delta, gamma, omega], name='Duffing')

    @classmethod
    def func_duffing(x, t, *p):
        x, y  = x
        alpha, beta, delta, gamma, omega = p
        dx = y
        dy = -delta * y - alpha * x - beta * x**3 + gamma * np.cos(omega * t)
        return [dx, dy]




# Clase Attractor: Permite representar un atractor de un sistema dinámico a partir de un vector de tiempo t y una matriz de trayectorias x.


class Trajectory:
    def __init__(self, t, x):
        self.x = x # Devuelve matriz de trayectorias x
        self.t = t # Devuelve vector de tiempo t
        self.n_var = np.shape(x)[1]
        self.cols = [f'x{i + 1}' for i in range(self.n_var)]

    def to_table(self, col_names):
        col_names = [f'x{i + 1}' for i in range(self.n_var)] if col_names is None else col_names
        trajectory_table = pd.DataFrame(self.x, columns=col_names)
        trajectory_table = trajectory_table.insert(0, 't', self.t, True)
        return trajectory_table

    def plot_trajectory3d(self, size=(5,5)):
        assert n_var == 3, f'Number of variables must be 3, instead got {n_var}'
        fig = plt.figure(figsize=size)
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(self.x[:, 0],
                self.x[:, 1],
                self.x[:, 2])
        ax.set_title('Lorenz Attractor')
        ax.set_xlabel('$x_{1}$')
        ax.set_ylabel('$x_{2}$')
        ax.set_zlabel('$x_{3}$')

    def plot_trajectory2d(self, variables=(0,1), size=(5,5)):
        assert n_var >= 2, f'Number of variables must be greater or equal to 2, instead got {n_var}'
        fig = plt.figure(figsize=size)
        ax.plot(self.x[:, variables[0]], self.x[:, variables[1]], 'k-')
        ax.set_title(f'$x_{variables[0] + 1}$ - x_{variables[1]}')
        ax.set_ylabel(f'$x_{variables[1]}$')
        ax.set_xlabel(f'$x_{variables[0]}$')

    def plot_x1t(self):
        fig, ax = plt.subplots(figsize=size)
        ax.plot(self.t, self.x[:, 0], '$x_{1}$', label='$x_{1}(t)$')
        ax.legend()
        ax.set_title('$x_{1} - t$')
        ax.set_xlabel('$t$')
        ax.set_ylabel('$x_{1}(t)$')

    def plot_x2t(self):
        assert n_var >= 2, f'Number of variables must be greater or equal to 2, instead got {n_var}'
        fig, ax = plt.subplots(figsize=size)
        ax.plot(self.t, self.x[:, 1], '$x_{2}$', label='$x_{2}(t)$')
        ax.legend()
        ax.set_title('$x_{2} - t$')
        ax.set_xlabel('$t$')
        ax.set_ylabel('$x_{2}(t)$')

    def plot_x3t(self):
        assert n_var == 3, f'Number of variables must be 3, instead got {n_var}'
        fig, ax = plt.subplots(figsize=size)
        ax.plot(self.t, self.x[:, 2], '$x_{3}$', label='$x_{3}(t)$')
        ax.legend()
        ax.set_title('$x_{3} - t$')
        ax.set_xlabel('$t$')
        ax.set_ylabel('$x_{3}(t)$')


class Poincare:
    def __init__(self, t, x):
        self.t = t
        self.x = x

    def _fit(self, a, plane, grade, axis):
        assert plane >= 1 and plane < 4
        assert axis >=1 and axis < 4

        t = np.delete(t, [-1, -2])

        x1a = np.delete(np.roll(self.x[:, axis - 1], 2), [0, 1])
        x1b = np.delete(np.roll(self.x[:, axis - 1], 1), [0, 1])
        x1c = np.delete(self.x[:, axis - 1], [0, 1])

        x1_slices = np.vstack((x1a, x1b, x1c))

        x2a = np.delete(np.roll(self.x[:, plane - 1], 2), [0, 1])
        x2b = np.delete(np.roll(self.x[:, plane - 1], 1), [0, 1])
        x2c = np.delete(self.x[:, plane - 1], [0, 1])

        x2_slices = np.vstack((x2a, x2b, x2c))

        x1 = x1_slices[:, (x1[0] < a) & (x1[1] > a)]
        x2 = x2_slices[:, (x1[0] < a) & (x1[1] > a)]
        t_map = self.t[(x1[0] < a) & (x1[1] > a)]

        x12 = np.vstack((x2, x1))

        def poly(v, grade_poly):
            return np.polyfit(v[3:6], v[0:3], grade_poly)

        x12_coeff = np.apply_along_axis(poly, 0, x12, grade)

        def apply_poly(p, a_value):
            return p[0] * a_value**2 + p[1] * a_value + p[2] if len(b) == 3 else p[0] * a_value + p[1]

        x2_fit = np.apply_along_axis(apply_poly, 0, x12_coeff, a)

        x21 = np.delete(x2_fit, -1)
        x22 = np.delete(x2_fit, 0)

        xmap = np.array([[x21], [x22]])

        return t_map, xmap, x2

    def z_map(self, a=0, axis=1, grade=1):
        plane = 3
        fit = _fit(a, plane, grade, axis)
        map_z = Map(fit, plane, axis)

    def y_map(self, a=0, axis=1, grade=1):
        plane = 2
        fit = _fit(a, plane, grade, axis)
        map_y = Map(fit, plane, axis)

    def x_map(self, a=0, axis=2, grade=1):
        plane = 1
        fit = _fit(a, plane, grade, axis)
        map_x = Map(fit, plane, axis)


class Map:
    def __init__(self, t, n, i, plane, axis):
        self.n0 = n[0]
        self.n1 = n[1]
        self.iterations = i
        self.t_iter = t
        self.plane = plane
        self.axis = axis

    def plot_cobweb(self):
        title = 'Z map' if self.plane == 3 else 'Y map' if self.plane == 2 else 'X map'
        xlabel = 'z(i)' if self.plane == 3 else 'y(i)' if self.plane == 2 else 'x(i)'
        ylabel = 'z(i+1)' if self.plane == 3 else 'y(i+1)' if self.plane == 2 else 'x(i+1)'
        fig, ax = plt.subplots(1, figsize=(7,7))
        ax.plot(self.n0, self.n1, 'r;')
        ax.plot(np.linspace(0, 800, 800), np.linspace(0, 800, 800), 'k-')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    def plot_iterations(self):
        title = 'Z(t)' if self.plane == 3 else 'Y(t)' if self.plane == 2 else 'X(t)'
        ylabel = 'y(i)' if plane == 3 and axis == 1 or plane == 1 and axis == 3 else 'x(i)' if plane == 3 and axis == 2 or plane == 2 and axis == 3 else 'z(i)' if plane == 2 and axis == 1 or plane == 1 and axis == 2 else None
        fig, ax = plt.subplots(1, figsize=(7,7))
        ax.plot(self.t_iter, self.iterations, 'k.')
        ax.set_title(title)
        ax.set_xlabel('n')
        ax.set_ylabel(ylabel)

