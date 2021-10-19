# Testing code


import caospy

import pandas as pd

import pytest


# Test for odeint  numerical integration
@pytest.fixture
# Test function dy/dt = y- y^2 to check numerical integration.
def fun():
    def dy(t, y, par):
        der = y - (y ** 2)
        return der

    f = caospy.Functional(dy, "Test Function")
    return f


def test_odeint(fun):
    y0 = [0.5]
    t0 = 0.0
    tf = 10
    par = [0]
    n = 500
    t, y = fun.time_evolution(y0, par, t0, tf, n)
    assert y[0, -1] == pytest.approx(1.000, rel=1e-3)


# Test for check zeroos values
@pytest.fixture
def lorenz():
    derivate = caospy.Lorenz()
    return derivate


def test_roots(lorenz):
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


# Tetst for check eigenvalues and eigenvectors
@pytest.fixture
def funtwodim():
    variables = ["x", "y"]
    param = ["a", "b"]
    functions = ["x+y", "a*x-b*y"]
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
    vec1 = 1 / vec[0, 0, 0] * vec[0, 0]
    vec2 = 1 / vec[0, 1, 0] * vec[0, 1]
    assert list(vec1) in ([1, 1], [1, -4])
    assert list(vec2) in ([1, 1], [1, -4])


# Tetst for check OneDimensional dataframe
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
    funciones1 = ["x**2 -1"]
    p1 = []
    s1 = caospy.OneDim(variables1, funciones1, parametros1, "A")
    l1 = s1.stability(p1)
    pd.testing.assert_frame_equal(l1, df)


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
# Test type function for numerical integration
def test_raisefunction():
    f = 4
    with pytest.raises(Exception) as exc:
        caospy.Functional(f, "check")
    assert (
        "The first argument must be a callable" + "got <class 'int'> instead."
    ) == str(exc.value)
