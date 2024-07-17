import firedrake as fd
from fireshape import ShapeObjective
import numpy as np


class CauchyRiemannConstraint(ShapeObjective):
    """Weakly enforce the Cauchy-Riemann equations"""

    def __init__(self, pde_solver, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pde_solver = pde_solver

    def value_form(self):
        """Evaluate misfit functional."""

        if self.pde_solver.failed_to_solve:  # return NaNs if state solve fails
            return np.nan * fd.dx(self.pde_solver.mesh_m)
        else:
            u, _ = fd.split(self.pde_solver.solution)
            Bu = fd.as_vector([fd.Dx(u[0], 0) - fd.Dx(u[1], 1), fd.Dx(u[1], 0) + fd.Dx(u[0], 1)])
            return fd.inner(Bu, Bu) * fd.dx