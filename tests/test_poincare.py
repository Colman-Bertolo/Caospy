# Code for test poincare maps

import numpy as np

import pytest


# Asset warnings
def test_map_differnetvar(lorenz):
    p = [10.0, 166.04, 8 / 3]
    x0 = [2.0, -1.0, 150.0]
    poincare_lorenz = lorenz.poincare(x0, p, t_disc=1, t_calc=2)
    with pytest.raises(ValueError):
        poincare_lorenz.map_v("x", "x")


def test_map_varisnotsystem(lorenz):
    p = [10.0, 166.04, 8 / 3]
    x0 = [2.0, -1.0, 150.0]
    poincare_lorenz = lorenz.poincare(x0, p, t_disc=0.3, t_calc=1)
    with pytest.raises(ValueError):
        poincare_lorenz.map_v("k", "x")


def test_map_varisoutsystem(lorenz):
    p = [10.0, 166.04, 8 / 3]
    x0 = [2.0, -1.0, 150.0]
    poincare_lorenz = lorenz.poincare(x0, p, t_disc=0.3, t_calc=1)
    with pytest.raises(ValueError):
        poincare_lorenz.map_v(6, "x")


def test_map_fixedisnotsystem(lorenz):
    p = [10.0, 166.04, 8 / 3]
    x0 = [2.0, -1.0, 150.0]
    poincare_lorenz = lorenz.poincare(x0, p, t_disc=0.3, t_calc=1)
    with pytest.raises(ValueError):
        poincare_lorenz.map_v("x", "k")


def test_map_fixedsoutsystem(lorenz):
    p = [10.0, 166.04, 8 / 3]
    x0 = [2.0, -1.0, 150.0]
    poincare_lorenz = lorenz.poincare(x0, p, t_disc=0.3, t_calc=1)
    with pytest.raises(ValueError):
        poincare_lorenz.map_v("x", 6)


# ----------------------------------------- end warnings tests
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


# Test for poincare  fit function
def test_fit(lorenz):
    sigma = 10
    rho = 166.04
    beta = 8 / 3
    par = [sigma, rho, beta]  # Parameters
    x0 = [2, -1, 150]  # Initials conditions
    lor = lorenz.poincare(x0, par, 500, 40)
    tc, mp, y12 = lor._fit(0, 1, 2, 2)
    tc_expected = np.array(
        [
            0.2680268,
            1.40014001,
            2.53625363,
            3.66836684,
            4.80048005,
            5.93259326,
            7.06470647,
            8.19681968,
            9.33293329,
            10.4650465,
            11.59715972,
            12.72927293,
            13.86138614,
            14.99749975,
            16.12961296,
            17.26172617,
            18.39383938,
            19.5259526,
            20.65806581,
            21.79417942,
            22.92629263,
            24.05840584,
            25.19051905,
            26.32263226,
            27.45874587,
            28.59085909,
            29.7229723,
            30.85508551,
            31.98719872,
            33.11931193,
            34.25542554,
            35.38753875,
            36.51965197,
            37.65176518,
            38.78387839,
            39.919992,
        ]
    )
    mp_expected = np.array(
        [
            [
                0.0,
                41.28472937,
                41.28540907,
                41.28580979,
                41.28475662,
                41.28435775,
                41.28449158,
                41.28502217,
                41.28580015,
                41.28521993,
                41.28449233,
                41.28436301,
                41.28470344,
                41.28537118,
                41.28587911,
                41.28479152,
                41.28436373,
                41.28447485,
                41.28498962,
                41.28575927,
                41.28527224,
                41.28451274,
                41.28435737,
                41.28467829,
                41.28533369,
                41.28595022,
                41.28482797,
                41.28437098,
                41.28445907,
                41.28495765,
                41.28571861,
                41.28532622,
                41.28453457,
                41.28435285,
                41.28465391,
                41.28529662,
            ],
            [
                0.0,
                41.28540907,
                41.28580979,
                41.28475662,
                41.28435775,
                41.28449158,
                41.28502217,
                41.28580015,
                41.28521993,
                41.28449233,
                41.28436301,
                41.28470344,
                41.28537118,
                41.28587911,
                41.28479152,
                41.28436373,
                41.28447485,
                41.28498962,
                41.28575927,
                41.28527224,
                41.28451274,
                41.28435737,
                41.28467829,
                41.28533369,
                41.28595022,
                41.28482797,
                41.28437098,
                41.28445907,
                41.28495765,
                41.28571861,
                41.28532622,
                41.28453457,
                41.28435285,
                41.28465391,
                41.28529662,
                41.28602313,
            ],
        ]
    )
    y12_expected = np.array(
        [
            41.28472937,
            41.28540907,
            41.28580979,
            41.28475662,
            41.28435775,
            41.28449158,
            41.28502217,
            41.28580015,
            41.28521993,
            41.28449233,
            41.28436301,
            41.28470344,
            41.28537118,
            41.28587911,
            41.28479152,
            41.28436373,
            41.28447485,
            41.28498962,
            41.28575927,
            41.28527224,
            41.28451274,
            41.28435737,
            41.28467829,
            41.28533369,
            41.28595022,
            41.28482797,
            41.28437098,
            41.28445907,
            41.28495765,
            41.28571861,
            41.28532622,
            41.28453457,
            41.28435285,
            41.28465391,
            41.28529662,
            41.28602313,
        ]
    )
    assert np.all((tc_expected - tc) < 1e-4)
    assert np.all((mp_expected - mp) < 1e-4)
    assert np.all((y12_expected - y12) < 1e-4)


# Test for poincare  map_v function
def test_map(lorenz):
    p = [10.0, 166.04, 8 / 3]
    x0 = [2.0, -1.0, 150.0]
    poincare_lorenz = lorenz.poincare(x0, p, t_disc=500, t_calc=40)
    z_map = poincare_lorenz.map_v("z", "x")
    value_test = np.array(
        [
            0,
            41.286087463433100,
            41.286084886611880,
            41.286078465187190,
            41.286071814277264,
            41.286064687685840,
            41.286057995322100,
            41.286052033471130,
            41.286048873520570,
            41.286047436280210,
            41.286085858964334,
            41.286079451427916,
            41.286078410378330,
            41.286053450083490,
            41.286058494775155,
            41.286066142012790,
            41.286075885288710,
            41.286084248813670,
            41.286056797634720,
            41.286063412820695,
            41.286072833550000,
            41.286081266025800,
            41.286054438991110,
            41.286047470994310,
            41.286048864767004,
            41.286047831805010,
            41.286062952575925,
            41.286056180382710,
            41.286050839775950,
            41.286048305566275,
            41.286047247330735,
            41.286058319695990,
            41.286052264980775,
            41.286048948907485,
            41.286047469604770,
            41.286085959637010,
            41.286079555278825,
        ]
    )
    value_test0 = value_test[0:-1]
    value_test1 = value_test[1::]
    assert np.all((z_map.n0 - value_test0) < 1e-2)
    assert np.all((z_map.n1 - value_test1) < 1e-2)
