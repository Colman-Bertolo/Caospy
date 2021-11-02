# Code for test poincare maps

import numpy as np

import pytest


# Test for poincare odeint map
def test_lorenz_maps(lorenz):
    sigma = 10
    rho = 166.04
    beta = 8 / 3
    par = [sigma, rho, beta]  # Parameters
    x0 = [2, -1, 150]  # Initials conditions
    lor = lorenz.poincare(x0, par, 500, 5)
    assert lor.x[:, -1] == pytest.approx(
        [11.8579, -9.1806, 159.9543], rel=1e-3
    )


# Test for check trajectories table
def test_lorenz_table(time_evolution_fun):
    y0 = [0.5]
    tdesc = 20.0
    tf = 10
    par = 0
    n = 1000
    states = time_evolution_fun.poincare(y0, par, tdesc, tf, n)
    table = states.to_table()
    assert np.all(table.columns[0] == "t")
