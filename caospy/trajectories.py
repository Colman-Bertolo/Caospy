# Systems trajectories treatment

import numpy as np
import pandas as pd
import sympy as sp

class Trajectory:
    def __init__(self, t, x, variables):
        self.x = x  # Devuelve matriz de trayectorias x
        self.t = t  # Devuelve vector de tiempo t
        self.n_var = np.shape(x)[1]
        self.cols = ["t"] + [f"{v}" for v in variables]

    def to_table(self):
        col_names = self.cols
        pd.set_option("display.precision", 2)
        merge_array = np.insert(self.x, 0, self.t, axis=1)
        trajectory_table = pd.DataFrame(merge_array, columns=col_names)
        return trajectory_table



