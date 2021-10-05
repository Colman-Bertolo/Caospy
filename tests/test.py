# Testing code


import caospy
import pytest


# Test for odeint  numerical integration
@pytest.fixture
# Test function dy/dt = y- y^2 to check numerical integration.
def fun():
    def dy(y, t, par):
        der = y-(y**2)
        return der
    f = caospy.Functional(dy, "Test Function")
    return f


def test_odeint(fun):
    y0 = 0.5
    t0 = 0.0
    tf = 10
    par = 0
    n = 500
    t, y = fun.time_evolution(y0, par, t0, tf, n)
    assert y[-1] == pytest.approx(1.000, rel=1e-3)


# Test for check zeroos values
@pytest.fixture
def lorenz():
    derivate = caospy.Lorenz()
    return derivate


def test_roots(lorenz):
    sigma = 10
    rho = 28
    beta = 8/3
    p = [sigma, rho, beta]
    j = lorenz.fixed_points(p)
    as1 = [0, 0, 0]
    as2 = [(beta*(rho-1))**0.5, (beta*(rho-1))**0.5, rho-1]
    as3 = [-(beta*(rho-1))**0.5, -(beta*(rho-1))**0.5, rho-1]
    assert j[0] in (as1, as2, as3)
    assert j[1] in (as1, as2, as3)
    assert j[2] in (as1, as2, as3)

# Tetst for check eigenvalues and eigenvectors
@pytest.fixture 
def funtwodim():
    variables = ['x', 'y']
    param = ['a', 'b']
    functions = ['x+y', 'a*x-b*y']
    derivate = caospy.TwoDim(variables, functions, param, 'Test 2D')
    return derivate

def test_eigenvalue(funtwodim):
    par = [4, 2]
    val = funtwodim.eigenvalues(par)
    assert val[0,0] in [2,-3]
    assert val[0,1] in [2,-3]




