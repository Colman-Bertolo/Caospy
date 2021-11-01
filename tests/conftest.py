import caospy
import pytest


@pytest.fixture
# Test function dy/dt = y- y^2 to check numerical integration.
def time_evolution_fun():
    def dy(t, y, par):
        der = y - (y ** 2)
        return der

    f = caospy.Functional(dy, "Test Function")
    return f


@pytest.fixture
# Test system, with ti > tf
def time_interval_fun():
    def derivative(t, y, par):
        der = y * par + y ** 2
        return der

    s = caospy.Functional(derivative, "Test time interval")
    return s


@pytest.fixture
def system0():
    var0 = ["x", "y"]
    func0 = ["4 * x - 3 * x * y", "7 * x**2 - 6 * y"]
    par0 = []
    test0 = caospy.TwoDim(var0, func0, par0, "A0")
    return test0


@pytest.fixture
def system1():
    var1 = ["x", "y"]
    func1 = ["4 * x - 3 * y", "8 * x - 6 * y"]
    par1 = []
    test1 = caospy.TwoDim(var1, func1, par1, "A1")
    return test1


@pytest.fixture
def system2():
    var2 = ["x", "y"]
    func2 = ["exp(x)", "8 * x - 6 * y"]
    par2 = []
    test2 = caospy.TwoDim(var2, func2, par2, "A2")
    return test2


@pytest.fixture
def system3():
    var3 = ["x", "y"]
    func3 = ["cos(x) - y", "x**2 - 6 * y"]
    par3 = []
    test3 = caospy.TwoDim(var3, func3, par3, "A3")
    return test3


@pytest.fixture
def system4():
    var4 = ["x"]
    func4 = ["cos(x) - x"]
    par4 = []
    test4 = caospy.OneDim(var4, func4, par4, "A4")
    return test4


@pytest.fixture
def system5():
    var5 = ["Q"]
    func5 = ["v0 / R - Q / (R * C)"]
    par5 = ["v0", "R", "C"]
    test5 = caospy.OneDim(var5, func5, par5, "A5")
    return test5


@pytest.fixture
def lorenz():
    derivate = caospy.Lorenz()
    return derivate


@pytest.fixture
def logistic():
    derivate = caospy.Logistic()
    return derivate


@pytest.fixture
def funtwodim():
    variables = ["x", "y"]
    param = ["a", "b"]
    functions = ["x + y", "a * x - b * y"]
    derivate = caospy.TwoDim(variables, functions, param, "Test 2D")
    return derivate


@pytest.fixture
def zero_slope_syst():
    v = ["x"]
    func = ["x**2 + 2 * x + 1"]
    par = []
    s = caospy.OneDim(v, func, par, "Linear Stability")
    return s
