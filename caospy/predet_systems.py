# Predetermined Dynamical Systems

from .core import *

# Lorenz's system

class Lorenz(MultiVarMixin, AutoSymbolic):
    _name = "Lorenz"
    _variables = ["x", "y", "z"]
    _parameters = ["sigma", "rho", "beta"]
    _functions = [
        "sigma * (y - x)",
        "x * (rho - z) - y",
        "x * y - beta * z"
    ]

# Duffing's system

class Duffing(AutoSymbolic):
    _name = "Duffing"
    _variables = ["x", "y", "t"]
    _parameters = ["alpha", "beta", "delta", "gamma", "omega"]
    _functions = [
        "y",
        "-delta * y - alpha * x - beta * x**3 + gamma * cos(omega * t)",
    ]

# Logistic system

class Logistic(OneDimMixin, AutoSymbolic):
    _name = "Logistic"
    _variables = ["N"]
    _parameters = ["r", "k"]
    _functions = ["r * N * (1 - N / k)"]

# Rossler-chaos system

class RosslerChaos(MultiVarMixin, AutoSymbolic):
    _name = "Rossler-chaos"
    _variables = ["x", "y", "z"]
    _parameters = ["a", "b", "c"]
    _functions = [
        "- (y + z)",
        "x + a * y",
        "b + z * (x - c)"
    ]

# Rossler-hyperchaos system

class RosslerChaos(MultiVarMixin, AutoSymbolic):
    _name = "Rossler-hyperchaos"
    _variables = ["x", "y", "z", "w"]
    _parameters = ["a", "b", "c", "d"]
    _functions = [
        "- (y + z)",
        "x + a * y + w",
        "b + x * z",
        "c * w - d * z"
    ]
