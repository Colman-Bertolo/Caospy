# Testing code for core.py


import caospy
import numpy as np
import pandas as pd
import pytest
import sympy as sp


# Test for Functional Class init - func type:
def test_callable_type_function():
    f = 1
    with pytest.raises(TypeError):
        caospy.Functional(f, "check")


# Test for Functional Class init - name type:
def test_str_type_name():
    def f():
        return 1

    with pytest.raises(TypeError):
        caospy.Functional(f, 3)


# Test for integration interval, tf > ti
def test_time_interval(time_interval_fun):
    y0 = [1]
    par = [2]
    t0 = 1.0
    tf = 0.0
    with pytest.raises(ValueError):
        time_interval_fun.time_evolution(y0, par, t0, tf)


# Tests for Symbolic class init

# Test for name type
def test_symbolic_name():
    v = ["x", "y", "z"]
    f = ["x**2 + c * y", "z - y * a", "x * b / z"]
    p = ["a", "b", "c"]
    with pytest.raises(TypeError):
        caospy.Symbolic(v, f, p, 2)


# Test for list type (parameters, variables, functions)
def test_list_type_vfp():
    f = ["x**2 + c * y", "z - y * a", "x * b / z"]
    p = ["a", "b", "c"]
    name = "testing_sys"
    with pytest.raises(TypeError):
        caospy.Symbolic(2, f, p, name)


# Test for str type in parameters, variables, functions
def test_str_type_vfp():
    v = ["x", "y", "z"]
    f = ["x**2 + c * y", "z - y * a", "x * b / z"]
    name = "testing_sys"
    with pytest.raises(TypeError):
        caospy.Symbolic(v, f, [1, 2], name)


# Test for shape of the system:
def test_system_shape():
    f = ["x**2 + c * y", "z - y * a", "x * b / z"]
    p = ["a", "b", "c"]
    name = "testing_sys"
    with pytest.raises(ValueError):
        caospy.Symbolic(["v"], f, p, name)


# Test for odeint  numerical integration


def test_time_evolution(time_evolution_fun):
    y0 = [0.5]
    t0 = 0.0
    tf = 10
    par = 0
    n = 1000
    states = time_evolution_fun.time_evolution(y0, par, t0, tf, n)
    y = states.x
    assert y[0][-1] == pytest.approx(1.000, rel=1e-3)


# Test for check zeros values


def test_roots_0(system0):
    p = []
    j = system0.fixed_points(p)
    as1 = np.array(
        [[0.0, 0.0], [-1.06904497, 1.33333333], [1.06904497, 1.3333333]]
    )
    assert np.all(j - as1 < 1e-5)


def test_roots_1(system1):
    p = []
    j = system1.fixed_points(p)
    y = sp.symbols("y")
    as1 = np.array([[3 * y / 4, y]], dtype=object)
    assert np.all(j - as1 < 1e-5)


def test_roots_2(system2):
    p = []
    j = system2.fixed_points(p)
    assert j is None


def test_roots_3(system3):
    p = []
    initial_guess = [1, 1]
    j = system3.fixed_points(p, initial_guess)
    as1 = np.array([[1.28983525, 0.27727916]])
    assert np.all(j - as1 < 1e-5)


def test_roots_4(system4):
    p = []
    j = system4.fixed_points(p)
    as1 = np.array([[0.73908513]])
    assert np.all(j - as1 < 1e-5)


def test_roots_5(system5):
    p = [20, 10, 0.02]
    j = system5.fixed_points(p)
    as1 = np.array([[0.4]])
    assert np.all(j - as1 < 1e-5)


def test_roots_lorenz(lorenz):
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


def test_roots_logistic(logistic):
    r = 2
    k = 4
    p = [r, k]
    j = logistic.fixed_points(p)
    as1 = [0, k]
    assert list(j) in (as1,)


# Tetst for check eigenvalues and eigenvectors


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


def test_onedim_zero_slope(zero_slope_syst):
    p = []
    with pytest.raises(caospy.LinearityError):
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


# Non-isolated
def test_nonisolated():
    variables = ["x", "y"]
    funciones = ["4*x-3*y", "8*x-6*y"]
    p1 = []
    s1 = caospy.TwoDim(variables, funciones, p1, "Non-isolated")
    l_center = s1.fixed_point_classify(p1)
    assert list(l_center["$Type$"]) == [
        "Non Isolated Fixed-Points," + "Line of Lyapunov stable fixed points"
    ]


# Tests for Poincare Map
# -------------------------------------------------------------------------
# Tests raises errors
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
