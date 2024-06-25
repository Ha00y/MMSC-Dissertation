import firedrake as fd
from fireshape import ShapeObjective
import numpy as np


class AerofoilObjective(ShapeObjective):
    """L2 tracking functional for Poisson problem."""

    def __init__(self, pde_solver, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pde_solver = pde_solver

    def value_form(self):
        """Evaluate misfit functional."""
        #nu = self.pde_solver.viscosity
        Re = self.pde_solver.Re

        if self.pde_solver.failed_to_solve:  # return NaNs if state solve fails
            return np.nan * fd.dx(self.pde_solver.mesh_m)
        else:
            z = self.pde_solver.solution
            u, p = fd.split(z)
            #return nu * fd.inner(fd.grad(u), fd.grad(u)) * fd.dx
            return 1/Re * fd.inner(fd.grad(u), fd.grad(u)) * fd.dx