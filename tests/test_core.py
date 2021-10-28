# Testing code for core.py


import caospy
import pytest
import pandas as pd
import numpy as np
import sympy as sp


# Test for Functional Class init - func type:
def test_callable_type_function():
    f = 1
    with pytest.raises(TypeError) as exc:
        caospy.Functional(f, "check")

# Test for Functional Class init - name type:
def test_str_type_name():
    def f():
        return 1
    with pytest.raises(TypeError) as exc:
        caospy.Functional(f, 3)

# Test for integration interval, tf > ti
@pytest.fixture
# Test system, with ti > tf
def time_interval_fun():
    def derivative(t, y, par):
        der = y * par + y**2
        return der

    s = caospy.Functional(derivative, "Test time interval")
    return s

def test_time_interval(time_interval_fun):
    y0 = [1]
    par = [2]
    t0 = 1.0
    tf = 0.0
    with pytest.raises(ValueError) as exc:
        states = time_interval_fun.time_evolution(y0, par, t0, tf)

# Tests for Symbolic class init

# Test for name type
def test_symbolic_name():
    v = ['x', 'y', 'z']
    f = ['x**2 + c * y', 'z - y * a', 'x * b / z']
    p = ['a', 'b', 'c']
    with pytest.raises(TypeError) as exc:
        caospy.Symbolic(v, f, p, 2)

# Test for list type (parameters, variables, functions)
def test_list_type_vfp():
    f = ['x**2 + c * y', 'z - y * a', 'x * b / z']
    p = ['a', 'b', 'c']
    name = 'testing_sys'
    with pytest.raises(TypeError) as exc:
        caospy.Symbolic(2, f, p, name)

# Test for str type in parameters, variables, functions
def test_str_type_vfp():
    v = ['x', 'y', 'z']
    f = ['x**2 + c * y', 'z - y * a', 'x * b / z']
    name = 'testing_sys'
    with pytest.raises(TypeError) as exc:
        caospy.Symbolic(v, f, [1, 2], name)

# Test for shape of the system:
def test_system_shape():
    f = ['x**2 + c * y', 'z - y * a', 'x * b / z']
    p = ['a', 'b', 'c']
    name = 'testing_sys'
    with pytest.raises(ValueError) as exc:
        caospy.Symbolic(['v'], f, p, name)

# Test for odeint  numerical integration
@pytest.fixture
# Test function dy/dt = y- y^2 to check numerical integration.
def time_evolution_fun():
    def dy(t, y, par):
        der = y - (y ** 2)
        return der

    f = caospy.Functional(dy, "Test Function")
    return f

def test_time_evolution(time_evolution_fun):
    y0 = [0.5]
    t0 = 0.0
    tf = 10
    par = 0
    n = 1000
    states = time_evolution_fun.time_evolution(y0, par, t0, tf)
    y = states.x
    assert y[0][-1] == pytest.approx(1.000, rel=1e-3)


# Test for check zeros values

@pytest.fixture
def system0():
    var0 = ['x', 'y']
    func0 = ['4 * x - 3 * x * y', '7 * x**2 - 6 * y']
    par0 = []
    Test0 = caospy.TwoDim(var0, func0, par0,'A0')
    return Test0

def test_roots_0(system0):
    p = []
    j = system0.fixed_points(p)
    as1 = np.array(
        [[0., 0.],
        [-1.06904497, 1.33333333],
        [1.06904497, 1.3333333]]
        )
    assert np.all(j - as1 < 1e-5)

@pytest.fixture
def system1():
    var1 = ['x', 'y']
    func1 = ['4 * x - 3 * y', '8 * x - 6 * y']
    par1 = []
    Test1 = caospy.TwoDim(var1, func1, par1,'A1')
    return Test1

def test_roots_1(system1):
    p = []
    j = system1.fixed_points(p)
    y = sp.symbols('y')
    as1 = np.array(
        [[3 * y / 4, y]],
        dtype=object
        )
    assert np.all(j - as1 < 1e-5)

@pytest.fixture
def system2():
    var2 = ['x', 'y']
    func2 = ['exp(x)', '8 * x - 6 * y']
    par2 = []
    Test2 = caospy.TwoDim(var2, func2, par2,'A2')
    return Test2

def test_roots_2(system2):
    p = []
    j = system2.fixed_points(p)
    assert j is None

@pytest.fixture
def system3():
    var3 = ['x', 'y']
    func3 = ['cos(x) - y', 'x**2 - 6 * y']
    par3 = []
    Test3 = caospy.TwoDim(var3, func3, par3,'A3')
    return Test3

def test_roots_3(system3):
    p = []
    initial_guess = [1, 1]
    j = system3.fixed_points(p, initial_guess)
    as1 = np.array(
        [[1.28983525, 0.27727916]]
        )
    assert np.all(j - as1 < 1e-5)

@pytest.fixture
def system4():
    var4 = ['x']
    func4 = ['cos(x) - x']
    par4 = []
    Test4 = caospy.OneDim(var4, func4, par4,'A4')
    return Test4

def test_roots_4(system4):
    p = []
    j = system4.fixed_points(p)
    as1 = np.array(
        [[0.73908513]]
        )
    assert np.all(j - as1 < 1e-5)

@pytest.fixture
def system5():
    var5 = ['Q']
    func5 = ['v0 / R - Q / (R * C)']
    par5 = ['v0', 'R', 'C']
    Test5 = caospy.OneDim(var5, func5, par5,'A5')
    return Test5

def test_roots_5(system5):
    p = [20, 10, 0.02]
    j = system5.fixed_points(p)
    as1 = np.array(
        [[0.4]]
        )
    assert np.all(j - as1 < 1e-5)

@pytest.fixture
def lorenz():
    derivate = caospy.Lorenz()
    return derivate

def test_roots_Lorenz(lorenz):
    sigma = 10
    rho = 28
    beta = 8 / 3
    p = [sigma, rho, beta]
    j = lorenz.fixed_points(p)
    as1 = [0, 0, 0]
    as2 = [(beta * (rho - 1)) ** 0.5, (beta * (rho - 1)) ** 0.5, rho - 1]
    as3 = [-((beta * (rho - 1)) ** 0.5), -((beta * (rho - 1)) ** 0.5), rho - 1]
    assert list(j[0]) in (as1, as2, as3)
    assert list(j[1]) in (as1, as2, as3)
    assert list(j[2]) in (as1, as2, as3)

@pytest.fixture
def logistic():
    derivate = caospy.Logistic()
    return derivate

def test_roots_Logistic(logistic):
    r = 2
    k = 4
    p = [r, k]
    j = logistic.fixed_points(p)
    as1 = [0, k]
    assert list(j) in (as1,)


# Tetst for check eigenvalues and eigenvectors

@pytest.fixture
def funtwodim():
    variables = ["x", "y"]
    param = ["a", "b"]
    functions = ["x + y", "a * x - b * y"]
    derivate = caospy.TwoDim(variables, functions, param, "Test 2D")
    return derivate


def test_eigenvalue(funtwodim):
    par = [4, 2]
    val = funtwodim.eigenvalues(par)
    assert val[0, 0] in [2, -3]
    assert val[0, 1] in [2, -3]


def test_eigenvectors(funtwodim):
    par = [4, 2]
    vec = funtwodim.eigenvectors(par)
    vec1 = 1 / vec[0, 0] * vec[0]
    vec2 = 1 / vec[1, 0] * vec[1]
    assert list(vec1) in ([1, 1], [1, -4])
    assert list(vec2) in ([1, 1], [1, -4])


# Tetst for check OneDim dataframe
def test_onedimdf():
    df = pd.DataFrame(
        {
            "Fixed Point": [[-1.0], [1.0]],
            "Slope": [-2.0, 2.0],
            "Stability": [True, False],
        }
    )
    variables1 = ["x"]
    parametros1 = []
    funciones1 = ["x**2 - 1"]
    p1 = []
    s1 = caospy.OneDim(variables1, funciones1, parametros1, "A")
    l1 = s1.stability(p1)
    pd.testing.assert_frame_equal(l1, df)

# Test for OneDim
@pytest.fixture
def zero_slope_syst():
    v = ['x']
    func = ['x**2 + 2 * x + 1']
    par = []
    s = caospy.OneDim(v, func, par, 'Linear Stability')
    return s


def test_onedim_zero_slope(zero_slope_syst):
    p = []
    with pytest.raises(caospy.LinearityError) as exc:
        zero_slope_syst.stability(p)


# ---------------------------------------
# Tetst for check TwoDimensional dataframes
# ---------------------------------------


# Saddle points
def test_saddle():
    variables = ["x", "y"]
    funciones = ["x+2*y", "3*x+4*y"]
    p1 = []
    s1 = caospy.TwoDim(variables, funciones, p1, "Saddle")
    l_saddle = s1.fixed_point_classify(p1)
    assert list(l_saddle["$Type$"]) == ["Saddle"]
    assert list(l_saddle["$\u0394$"]) == [-2 + 0j]


# Unstable Node
def test_node():
    variables = ["x", "y"]
    funciones = ["2*x+y", "3*x+4*y"]
    p1 = []
    s1 = caospy.TwoDim(variables, funciones, p1, "Saddle")
    l_node = s1.fixed_point_classify(p1)
    assert list(l_node["$Type$"]) == ["Unstable Node"]
    assert list(l_node["$\u0394$"]) == [5 + 0j]


# Stable Node
def test_stable():
    variables = ["x", "y"]
    funciones = ["y", "-2*x-3*y"]
    p1 = []
    s1 = caospy.TwoDim(variables, funciones, p1, "Stable")
    l_stable = s1.fixed_point_classify(p1)
    assert list(l_stable["$Type$"]) == ["Stable Node"]


# Degenerate Node
def test_degenerate():
    variables = ["x", "y"]
    funciones = ["3*x-4*y", "x-y"]
    p1 = []
    s1 = caospy.TwoDim(variables, funciones, p1, "Degenerate")
    l_degenerate = s1.fixed_point_classify(p1)
    assert list(l_degenerate["$Type$"]) == ["Unstable Degenerate Node"]


# Center Node
def test_center():
    variables = ["x", "y"]
    funciones = ["5*x+2*y", "-17*x-5*y"]
    p1 = []
    s1 = caospy.TwoDim(variables, funciones, p1, "Center")
    l_center = s1.fixed_point_classify(p1)
    assert list(l_center["$Type$"]) == ["Center"]


