# ======================================================================
# IMPORTS
# ======================================================================

import caospy

from matplotlib.testing.decorators import check_figures_equal

import numpy as np

# ======================================================================


# Test plot for 1D function in trajectories file
@check_figures_equal(extensions=["png"])
def test_trajectorieplot_1D(fig_test, fig_ref):
    # Make the function to plot
    def dy(t, y, par):
        return par * y

    s = caospy.Functional(dy, "Test plot 1D", ["y"])
    par = 2
    y0 = [0.5]
    t0 = 0
    tf = 3
    N = 500
    fun = s.time_evolution(y0, par, t0, tf, N)  # Get Trajectorie class

    # test
    test_ax = fig_test.subplots()
    fun.plot_trajectory("t-y", ax=test_ax)

    # expected
    exp_ax = fig_ref.subplots()
    exp_ax.plot(fun.t, fun.x[0, :], "r", lw=2)
    exp_ax.grid()


# Test plot for 2D function in trajectories file
@check_figures_equal(extensions=["png"])
def test_trajectorieplot_2D(fig_test, fig_ref):
    # Make the function to plot
    def dy(t, y, par):
        return [y[1], -np.sin(y[0])]  # Pendulum Function

    var_name = ["r", "theta"]
    s = caospy.Functional(dy, "Test plot 2D", var_name)
    par = 0
    y0 = [0.1, 0]
    t0 = 0
    tf = 3
    N = 500
    fun = s.time_evolution(y0, par, t0, tf, N)  # Get Trajectorie class

    # test
    test_ax = fig_test.subplots()
    fun.plot_trajectory("r-theta", ax=test_ax)

    # expected
    exp_ax = fig_ref.subplots()
    exp_ax.plot(fun.x[0, :], fun.x[1, :], "r", lw=2)
    exp_ax.grid()


# Test for cobweb plot in poincare file
@check_figures_equal(extensions=["png"])
def test_plot_cobwebpoincare(fig_test, fig_ref):
    lorenz = caospy.Lorenz()
    p = [10.0, 166.04, 8 / 3]
    x0 = [2.0, -1.0, 150.0]
    poincare_lorenz = lorenz.poincare(x0, p, t_disc=60, t_calc=30)
    z_map = poincare_lorenz.map_v("z", "x")
    # test
    test_ax = fig_test.subplots()
    z_map.plot_cobweb(test_ax)
    # expected
    n0 = z_map.n0[z_map.n0 != 0]
    n1 = z_map.n1[z_map.n1 != 0]
    exp_ax = fig_ref.subplots()
    exp_ax.scatter(n0, n1, s=2, color="r")
    exp_ax.plot(
        np.linspace(min(n0), max(n0), 100),
        np.linspace(min(n1), max(n1), 100),
        color="k",
    )
    exp_ax.ticklabel_format(axis="both", style="plain", useOffset=False)


# Test for plot iterations
@check_figures_equal(extensions=["png"])
def test_plot_iterations(fig_test, fig_ref):
    lorenz = caospy.Lorenz()
    p = [10.0, 166.04, 8 / 3]
    x0 = [2.0, -1.0, 150.0]
    poincare_lorenz = lorenz.poincare(x0, p, t_disc=60, t_calc=30)
    z_map = poincare_lorenz.map_v(2, 0)
    # test
    test_ax = fig_test.subplots()
    z_map.plot_iterations(test_ax)
    # expected
    exp_ax = fig_ref.subplots()
    exp_ax.plot(
        z_map.t_iter,
        z_map.iterations,
        color="k",
        marker=".",
    )
