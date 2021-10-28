# Poincaré methods implementations

import numpy as np
import pandas as pd
import sympy as sp

from .trajectories import *

class Poincare(Trajectory):
    def __init__(self, t, x, variables):
        super().__init__(t, x, variables)

    def _fit(self, a, plane, grade, axis):
        if not plane >= 1 and plane < 4:
            raise ValueError("Specified plane must be between 1 and 3")
        if not axis >= 1 and axis < 4:
            raise ValueError("Specified axis must be between 1 and 3")

        t = np.delete(self.t, [-1, -2])

        x1a = np.delete(np.roll(self.x[:, axis - 1], 2), [0, 1])
        x1b = np.delete(np.roll(self.x[:, axis - 1], 1), [0, 1])
        x1c = np.delete(self.x[:, axis - 1], [0, 1])

        x1_slices = np.vstack((x1a, x1b, x1c))

        x2a = np.delete(np.roll(self.x[:, plane - 1], 2), [0, 1])
        x2b = np.delete(np.roll(self.x[:, plane - 1], 1), [0, 1])
        x2c = np.delete(self.x[:, plane - 1], [0, 1])

        x2_slices = np.vstack((x2a, x2b, x2c))

        x1 = x1_slices[:, (x1_slices[0] < a) & (x1_slices[1] > a)]
        x2 = x2_slices[:, (x1_slices[0] < a) & (x1_slices[1] > a)]
        t_map = t[(x1_slices[0] < a) & (x1_slices[1] > a)]

        x12 = np.vstack((x2, x1))

        def poly(v, grade_poly):
            return np.polyfit(v[3:6], v[0:3], grade_poly)

        x12_coeff = np.apply_along_axis(poly, 0, x12, grade)

        def apply_poly(p, a_value):
            return (
                p[0] * a_value ** 2 + p[1] * a_value + p[2]
                if len(p) == 3
                else p[0] * a_value + p[1]
            )

        x2_fit = np.apply_along_axis(apply_poly, 0, x12_coeff, a)

        x21 = np.delete(x2_fit, -1)
        x22 = np.delete(x2_fit, 0)

        xmap = np.array([[x21], [x22]])

        return t_map, xmap, x2


class Map:
    def __init__(self, t, n, i, plane, axis):
        self.n0 = n[0]
        self.n1 = n[1]
        self.iterations = i
        self.t_iter = t
        self.plane = plane
        self.axis = axis
