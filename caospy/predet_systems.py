# ==============================================================================
# Docs
# ==============================================================================

"""Set of predefined dynamical systems that are of common analysis."""

# ==============================================================================
# Imports
# ==============================================================================

from .core import AutoSymbolic, MultiVarMixin, OneDimMixin

# ==============================================================================
# Class Lorenz
# ==============================================================================


class Lorenz(MultiVarMixin, AutoSymbolic):
    """Implementation for Lorenz's system defined by the following equations.

    dx/dt = sigma (y - x)
    dy/dt = x (rho - z) - y
    dz/dt = x y - beta z

    The system has the same attributes as Symbolic and Functional types.

    Example
    -------
    >>> sample_Lorenz = caospy.Lorenz()
    >>> sample_Lorenz.variables
    ['x', 'y', 'z']
    >>> sample_Lorenz.f
    ['sigma * (y - x)', 'x * (rho - z) - y', 'x * y - beta * z']
    >>> sample_Lorenz.params
    ['sigma', 'rho', 'beta']
    """

    _name = "Lorenz"
    _variables = ["x", "y", "z"]
    _parameters = ["sigma", "rho", "beta"]
    _functions = ["sigma * (y - x)", "x * (rho - z) - y", "x * y - beta * z"]


# ==============================================================================
# Class Duffing
# ==============================================================================


class Duffing(AutoSymbolic):
    """Implementation for Duffing's system defined by the following equations.

    dx/dt = y
    dy/dt = -delta y - alpha x - beta x^3 + gamma cos(omega t)
    dt/dt = 1

    The system has the same attributes as Symbolic and Functional types.
    """

    _name = "Duffing"
    _variables = ["x", "y", "t"]
    _parameters = ["alpha", "beta", "delta", "gamma", "omega"]
    _functions = [
        "y",
        "-delta * y - alpha * x - beta * x**3 + gamma * cos(omega * t)",
    ]


# ==============================================================================
# Class Logistic
# ==============================================================================


class Logistic(OneDimMixin, AutoSymbolic):
    """Implementation for Logistic system defined by the following equation.

    dx/dt = r N (1 - N / k)

    The system has the same attributes as Symbolic and Functional types.

    Example
    -------
    >>> sample_logistic = caospy.Logistic()
    >>> sample_logistic.variables
    ['N']
    >>> sample_logistic.f
    ['r * N * (1 - N / k)']
    >>> sample_logistic.params
    ['r', 'k']
    """

    _name = "Logistic"
    _variables = ["N"]
    _parameters = ["r", "k"]
    _functions = ["r * N * (1 - N / k)"]


# ==============================================================================
# Class Rossler-Chaos
# ==============================================================================


class RosslerChaos(MultiVarMixin, AutoSymbolic):
    """Implementation for Rossler's (Chaos) system defined by the following equations.

    dx/dt = - (y + z)
    dy/dt = x + a y
    dz/dt = b + z (x - c)

    The system has the same attributes as Symbolic and Functional types.

    Example
    -------
    >>> sample_RosslerChaos = caospy.RosslerChaos()
    >>> sample_RosslerChaos.variables
    ['x', 'y', 'z']
    >>> sample_RosslerChaos.f
    ['- (y + z)', 'x + a * y', 'b + z * (x - c)']
    >>> sample_RosslerChaos.params
    ['a', 'b', 'c']
    """

    _name = "Rossler-chaos"
    _variables = ["x", "y", "z"]
    _parameters = ["a", "b", "c"]
    _functions = ["- (y + z)", "x + a * y", "b + z * (x - c)"]


# ==============================================================================
# Class Rossler - Hyper Chaos
# ==============================================================================


class RosslerHyperChaos(MultiVarMixin, AutoSymbolic):
    """Implementation for Rossler's (Hyper Chaos) system defined by the following equations.

    dx/dt = - (y + z)
    dy/dt = x + a y + w
    dz/dt = b + x z
    dw/dt = c w - d z

    The system has the same attributes as Symbolic and Functional types.

    Example
    -------
    >>> sample_RosslerHyperChaos = caospy.RosslerHyperChaos()
    >>> sample_RosslerHyperChaos.variables
    ['x', 'y', 'z', 'w']
    >>> sample_RosslerHyperChaos.f
    ['- (y + z)', 'x + a * y + w', 'b + x * z', 'c * w - d * z']
    >>> sample_RosslerHyperChaos.params
    ['a', 'b', 'c', 'd']
    """

    _name = "Rossler-hyperchaos"
    _variables = ["x", "y", "z", "w"]
    _parameters = ["a", "b", "c", "d"]
    _functions = ["- (y + z)", "x + a * y + w", "b + x * z", "c * w - d * z"]
