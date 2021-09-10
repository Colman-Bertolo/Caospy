import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sympy as sp
from scipy.optimize import fsolve


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

    def eigenvectors(self):


'''

def duffing(t, x, *c):
    x, y  = x
    alpha, beta, delta, gamma, omega = c
    dx = y
    dy = -delta * y - alpha * x - beta * x**3 + gamma * np.cos(omega * t)
    return [dx, dy]

def laser_plasma(t, x, *c):
    x1, x2, x3 = x
    n0, a0 = c
    h = 1 if x1 > 0 else 0
    bz = -np.sign(x3) * np.sqrt(2 * n0) * np.sqrt((1 + x3**2)**(1.5) - (1 + x3**2)) if x1 > 0 else '''



'''
Clase Lorenz: Permite representar un sistema de Lorenz de 3x3. Se inicializa con condiciones iniciales x0, y0, z0 y parámetros sigma, rho y beta.

'''

class Lorenz:
    def __init__(self, x0, y0, z0, sigma, rho, beta):
        self.x0 = x0 # Devuelve condición inicial x0
        self.y0 = y0 # Devuelve condición inicial y0
        self.z0 = z0 # Devuelve condición inicial z0
        self.X0 = [x0, y0, z0] # Devuelve lista de condiciones iniciales X0
        self.sigma = sigma # Devuelve parámetro sigma
        self.rho = rho # Devuelve parámetro rho
        self.beta = beta # Devuelve parámetro beta
        self.params = [sigma, rho, beta] # Devuelve lista de parámetros params
        self.params_tuple = (sigma, rho, beta) # Devuelve tupla de parámetros


    def attractor(self, step=0.01, n=5000):

        def lorenz(t, x, sigma, rho, beta):
            x, y, z = x
            sigma, rho, beta = c
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z
            return [dx, dy, dz]
        t = np.linspace(0, step * n, n)

        f = odeint(lorenz, self.X0, t, args=self.params_tuple, tfirst=True)
        f = Attractor(t, f)
        return f


    def poincare(self, t_desc=20000, t_calc=50, step=0.01):
        t_descarte = np.linspace(0, t_desc, int(t_desc / step))
        x = odeint(lorenz, self.X0, t_desc, self.params_tuple, tfirst=True)
        t_calculo = np.linspace(0, t_calc, int(t_calc / step))
        x0 = [x[0, -1], x[1, -1], x[2, -1]]
        x = odeint(lorenz, x0, t_calculo, self.params_tuple, tfirst=True)
        p = Poincare(t_calculo, x)
        return p


'''
Clase Duffing: Permite representar sistema de Duffing. Se inicializa a partir de condiciones iniciales x0 e y0 y parámetros alpha, beta, delta, gamma y omega.
'''


class Duffing:
    def __init__(self, x0, y0, alpha, beta, delta, gamma, omega):
        self.x0 = x0 # Devuelve condición inicial x0
        self.y0 = y0 # Devuelve condición inicial y0
        self.X0 = [x0, y0] # Devuelve condición inicial z0
        self.alpha = alpha # Devuelve parámetro alpha
        self.beta = beta # Devuelve parámetro beta
        self.delta = delta # Devuelve parámetro delta
        self.gamma = gamma # Devuelve parámetro gamma
        self.omega = omega # Devuelve parámetro omega
        self.params = [alpha, beta, delta, gamma, omega] # Devuelve lista de parámetros params
        self.params_tuple = (alpha, beta, delta, gamma, omega) # Devuelve tupla de parámetros params_tuple





# Clase Attractor: Permite representar un atractor de un sistema dinámico a partir de un vector de tiempo t y una matriz de trayectorias x.


class Attractor:
    def __init__(self, t, x):
        self.x = x # Devuelve matriz de trayectorias x
        self.t = t # Devuelve vector de tiempo t

    def plot_attractor(self, size=(7,7)):
        fig = plt.figure(figsize=size)
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(self.x[:, 0],
                self.x[:, 1],
                self.x[:, 2])
        ax.set_title('Lorenz Attractor')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    def plot_xt(self):
        fig, ax = plt.subplots(figsize=size)
        ax.plot(self.t, self.x[:, 0], 'x', label='x(t)')
        ax.legend()
        ax.set_title('x - t')
        ax.set_xlabel('t')
        ax.set_ylabel('x(t)')

    def plot_yt(self):
        fig, ax = plt.subplots(figsize=size)
        ax.plot(self.t, self.x[:, 1], 'y', label='y(t)')
        ax.legend()
        ax.set_title('y - t')
        ax.set_xlabel('t')
        ax.set_ylabel('y(t)')

    def plot_zt(self):
        fig, ax = plt.subplots(figsize=size)
        ax.plot(self.t, self.x[:, 2], 'z', label='z(t)')
        ax.legend()
        ax.set_title('z - t')
        ax.set_xlabel('t')
        ax.set_ylabel('z(t)')




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
        ylabel = 'y(i)' if plane == 3 and axis == 1 or plane == 1 and axis == 3 else 'x(i)' if plane == 3 and axis == 2 or plane == 2 and axis == 3 else 'z(i)' if plane == 2 and axis == 1 or plane == 1 and axis == 2
        fig, ax = plt.subplots(1, figsize=(7,7))
        ax.plot(self.t_iter, self.iterations, 'k.')
        ax.set_title(title)
        ax.set_xlabel('n')
        ax.set_ylabel(ylabel)

