# Testing code for core.py


import caospy

import numpy as np

import pandas as pd

import pytest

import sympy as sp


# ---------- Warning tests
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


# Test dimennsion OneDim
def test_onedim():
    f = ["x+y", "x*y"]
    v = ["x", "y"]
    par = []
    name = "onedim"
    with pytest.raises(Exception):
        caospy.OneDim(v, f, par, name)


# Test dimennsion TwoDim
def test_twodim():
    f = ["x+y"]
    v = ["x"]
    par = []
    name = "twodim"
    with pytest.raises(ValueError):
        caospy.TwoDim(v, f, par, name)


# End Warnings tests -------------------------------------------


# Numerical integration test
def test_time_evolution(time_evolution_fun):
    y0 = [0.5]
    t0 = 0.0
    tf = 10
    par = 0
    n = 1000
    states = time_evolution_fun.time_evolution(y0, par, ti=t0, tf=tf, n=n)
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
    assert np.all(j[0] == as11 for as11 in (as1, as2, as3))
    assert np.all(j[1] == as22 for as22 in (as1, as2, as3))
    assert np.all(j[2] == as33 for as33 in (as1, as2, as3))


def test_roots_logistic(logistic):
    r = 2
    k = 4
    p = [r, k]
    j = logistic.fixed_points(p)
    as1 = [[0], [k]]
    assert np.all(j == as1)


# Tetst for check eigenvalues and eigenvectors
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


# Test for checch full return amtrix
def test_reach4(reachfull):
    par = [45]
    matrices = reachfull.full_linearize(par)
    assert np.all(matrices[0] == [0])
    assert np.all(matrices[1] == par)
    assert np.all(matrices[2] == par)
    assert np.all(matrices[3] == [1])


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
    assert np.all(l_saddle["$Type$"] == ["Saddle"])
    assert np.all(l_saddle["$\u0394$"] == [-2 + 0j])


# Unstable Node
def test_node():
    variables = ["x", "y"]
    funciones = ["2*x+y", "3*x+4*y"]
    p1 = []
    s1 = caospy.TwoDim(variables, funciones, p1, "Saddle")
    l_node = s1.fixed_point_classify(p1)
    assert np.all(l_node["$Type$"] == ["Unstable Node"])
    assert np.all(l_node["$\u0394$"] == [5 + 0j])


# Stable Node
def test_stable():
    variables = ["x", "y"]
    funciones = ["y", "-2*x-3*y"]
    p1 = []
    s1 = caospy.TwoDim(variables, funciones, p1, "Stable")
    l_stable = s1.fixed_point_classify(p1)
    assert np.all(l_stable["$Type$"] == ["Stable Node"])


# Degenerate Node
def test_degenerate():
    variables = ["x", "y"]
    funciones = ["3*x-4*y", "x-y"]
    p1 = []
    s1 = caospy.TwoDim(variables, funciones, p1, "Degenerate")
    l_degenerate = s1.fixed_point_classify(p1)
    assert np.all(l_degenerate["$Type$"] == ["Unstable Degenerate Node"])


# Center Node
def test_center():
    variables = ["x", "y"]
    funciones = ["5*x+2*y", "-17*x-5*y"]
    p1 = []
    s1 = caospy.TwoDim(variables, funciones, p1, "Center")
    l_center = s1.fixed_point_classify(p1)
    assert np.all(l_center["$Type$"] == ["Center"])


# Non-isolated Lyapunov stable
def test_nonisolated():
    variables = ["x", "y"]
    funciones = ["4*x-3*y", "8*x-6*y"]
    p1 = []
    s1 = caospy.TwoDim(variables, funciones, p1, "Non-isolated")
    l_center = s1.fixed_point_classify(p1)
    assert np.all(
        l_center["$Type$"]
        == [
            "Non Isolated Fixed-Points,"
            + "Line of Lyapunov stable fixed points"
        ]
    )


# Non-isolated plane of fixed point
def test_nonisolated_planefxp():
    variables = ["x", "y"]
    funciones = ["y", "x**2"]
    p1 = []
    s1 = caospy.TwoDim(variables, funciones, p1, "Non-isolated")
    l_nipfxp = s1.fixed_point_classify(p1)
    assert np.all(
        l_nipfxp["$Type$"]
        == ["Non Isolated Fixed-Points," + "Plane of fixed points"]
    )


# Non-isolated Line of unstable fixed points
def test_nonisolated_lineunstfxp():
    variables = ["x", "y"]
    funciones = ["-x**2", "4*y-6"]
    p1 = []
    s1 = caospy.TwoDim(variables, funciones, p1, "Non-isolated")
    l_nipfxp = s1.fixed_point_classify(p1)
    assert np.all(
        l_nipfxp["$Type$"]
        == ["Non Isolated Fixed-Points," + "Line of unstable fixed points."]
    )


# Stable degenerate-node
def test_Stabledegeneratenode():
    variables = ["x", "y"]
    funciones = ["-x", "x-y"]
    p1 = []
    s1 = caospy.TwoDim(variables, funciones, p1, "stable_degnode")
    l_center = s1.fixed_point_classify(p1)
    assert np.all(l_center["$Type$"] == ["Stable Degenerate Node"])


# Test for stable and unstable star node
def test_node_stunst():
    variables = ["x", "y"]
    funciones = ["cos(x)", "sin(y)"]
    p1 = []
    s1 = caospy.TwoDim(variables, funciones, p1, "stable_degnode")
    l_node = s1.fixed_point_classify(p1)
    assert np.all(l_node["$Type$"][1] == "Stable Star Node")
    assert np.all(l_node["$Type$"][2] == "Unstable Star Node")


# Test Stable spiral fixed point
def test_Stablespriral():
    variables = ["x", "y"]
    funciones = ["y", "-sin(x)-1*y"]
    p1 = []
    spiral = caospy.TwoDim(variables, funciones, p1, "stable_spiral")
    l_spiral = spiral.fixed_point_classify(p1)
    assert np.all(l_spiral["$Type$"][0] == "Stable Spiral")


# Test Unstable spiral fixed point
def test_Unsttablespriral():
    variables = ["x", "y"]
    funciones = ["x", "sin(y)+8"]
    p1 = []
    spiral = caospy.TwoDim(variables, funciones, p1, "stable_spiral")
    l_spiraluns = spiral.fixed_point_classify(p1)
    assert np.all(l_spiraluns["$Type$"][1] == "Unstable Spiral")


# Test for check there's not fixed point onedim
def test_nofixedpoints1D():
    fun = ["2.74**x"]
    par = []
    var = ["x"]
    no_fixed = caospy.OneDim(var, fun, par, "No Fixed Points")
    point = no_fixed.stability(par)
    assert point == "There are no fixed points to evaluate"


# Test for check there's not fixed point twodim
def test_nofixedpoints2D():
    fun = ["2.74**x", "2.74**y"]
    par = []
    var = ["x", "y"]
    no_fixed_2d = caospy.TwoDim(var, fun, par, "No Fixed Points")
    point_2d = no_fixed_2d.fixed_point_classify(par)
    assert point_2d == "There is no fixed points to evaluate."
