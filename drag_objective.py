import firedrake as fd
from fireshape import ShapeObjective
import numpy as np


class DragObjective(ShapeObjective):
    """L2 tracking functional for Poisson problem."""

    def __init__(self, pde_solver, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pde_solver = pde_solver

    def value_form(self):
        """Evaluate misfit functional."""
        #Re = self.pde_solver.Re

        if self.pde_solver.failed_to_solve:  # return NaNs if state solve fails
            return np.nan * fd.dx(self.pde_solver.mesh)
        else:
            u, _ = fd.split(self.pde_solver.solution)
            return fd.inner(fd.sym(fd.grad(u)), fd.sym(fd.grad(u))) * fd.dx 