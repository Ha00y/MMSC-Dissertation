import firedrake as fd
from fireshape import ShapeObjective
import numpy as np

class LiftObjective(ShapeObjective):
    """Lift force functional for an aerofoil."""

    def __init__(self, pde_solver, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pde_solver = pde_solver

    def value_form(self):
        """Evaluate lift force functional."""
        if self.pde_solver.failed_to_solve:  # return NaNs if state solve fails
            return np.nan * fd.dx(self.pde_solver.mesh)
        else:
            u, p = fd.split(self.pde_solver.solution)  # Assuming solution is (velocity, pressure)
            n = fd.FacetNormal(self.pde_solver.mesh)  # Normal vector on the surface

            # Compute the pressure component of the lift
            pressure_lift = fd.inner(-p*n, fd.Constant([0,1])) * fd.ds(5)  # n[1] is the y-component of the normal

            # Assuming viscosity is available from the solver
            Re = self.pde_solver.Re

            # Compute the shear stress component of the lift
            shear_lift = 1/Re * fd.inner(2*fd.sym(fd.grad(u)) * n, fd.Constant([0,1])) * fd.ds(5)  

            return -(pressure_lift + shear_lift)

