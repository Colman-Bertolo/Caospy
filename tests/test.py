# Testing code


import caospy
import pytest


### Test for odeint  numerical integration
@pytest.fixture
# Test function dy/dt = y- y^2 to check numerical integration.
def fun():
    def dy(y, t, par):
        dy = y - y**2
        return dy
    f = caospy.Functional(dy, "Test Function")
    return f


def test_odeint(fun):
    y0 = 0.5
    t0 = 0.0
    tf = 10
    par = 0
    n = 500
    t, y = fun.time_evolution(y0, t0, tf, n, par)
    assert y[-1] == pytest.approx(1.000, rel=1e-3)
