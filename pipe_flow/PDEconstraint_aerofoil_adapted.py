import firedrake as fd
from fireshape import PdeConstraint
import numpy as np
import netgen
from netgen.occ import *

class NavierStokesSolver(PdeConstraint):
    """Incompressible Navier-Stokes as PDE constraint."""

    def __init__(self, mesh_m, viscosity):
        super().__init__()
        self.mesh_m = mesh_m
        self.failed_to_solve = False  # when self.solver.solve() fail

        # Setup problem, Taylor-Hood finite elements
        self.V = fd.VectorFunctionSpace(self.mesh_m, "CG", 4) \
            * fd.FunctionSpace(self.mesh_m, "DG", 3)

        # Preallocate solution variables for state equation
        self.solution = fd.Function(self.V, name="State")
        self.testfunction = fd.TestFunction(self.V)

        # Define viscosity parameter
        self.viscosity = viscosity

        n = fd.FacetNormal(self.mesh_m)
        (x, y) = fd.SpatialCoordinate(self.mesh_m)
        p0 = 10/13 - x/13 #1atleft,0atright

        # Weak form of incompressible Navier-Stokes equations
        z = self.solution
        u, p = fd.split(z)
        test = self.testfunction
        v, q = fd.split(test)
        nu = self.viscosity  # shorten notation
        self.F = nu*fd.inner(fd.grad(u), fd.grad(v))*fd.dx - p*fd.div(v)*fd.dx\
            + fd.inner(fd.dot(fd.grad(u), u), v)*fd.dx + fd.div(u)*q*fd.dx + fd.inner(p0*n, v)*fd.ds

        # Dirichlet Boundary conditions
        self.bcs = fd.DirichletBC(self.V.sub(0), fd.Constant((0,0)), (1,4,5))

        # PDE-solver parameters
        self.nsp = None
        self.sp = {
                    'snes_monitor': None,
                    'snes_converged_reason': None,
                    'snes_max_it': 20,
                    'snes_atol': 1e-8,
                    'snes_rtol': 1e-12,
                    'snes_stol': 1e-06,
                    'ksp_type': 'preonly',
                    'pc_type': 'lu',
                    'pc_factor_mat_solver_type': 'mumps'
                    }

    def solve(self):
        super().solve()
        self.failed_to_solve = False
        u_old = self.solution.copy(deepcopy=True)
        try:
            fd.solve(self.F == 0, self.solution, bcs=self.bcs,
                     solver_parameters=self.sp)
        except fd.ConvergenceError:
            self.failed_to_solve = True
            self.solution.assign(u_old)


if __name__ == "__main__":

    t = 0.12 # specify NACA00xx type
    viscosity = fd.Constant(0.1)

    N_x = 100
    x = np.linspace(0,1.0089,N_x)

    def naca00xx(x,t):
        y = 5*t*(0.2969*(x**0.5) - 0.1260*x - 0.3516*(x**2) + 0.2843*(x**3) - 0.1015*(x**4))
        return np.concatenate((x,np.flip(x)),axis=None), np.concatenate((y,np.flip(-y)),axis=None)

    x, y = naca00xx(x,t)

    pnts = [Pnt(x[i], y[i], 0) for i in range(len(x))]

    spline = SplineApproximation(pnts)
    aerofoil = Face(Wire(spline)).Move((0.3,1,0))
    rect = WorkPlane(Axes((-1, 0, 0), n=Z, h=X)).Rectangle(4, 2).Face()
    domain = rect - aerofoil

    domain.edges.name="wing"
    domain.edges.Min(Y).name="bottom"
    domain.edges.Max(Y).name="top"
    domain.edges.Min(X).name="inlet"
    domain.edges.Max(X).name="outlet"
    geo = OCCGeometry(domain, dim=2)

    ngmesh = geo.GenerateMesh(maxh=1)
    ngsolve_mesh = fd.Mesh(ngmesh)

    mh = fd.MeshHierarchy(ngsolve_mesh, 2)
    mesh = mh[-1]

    e = NavierStokesSolver(mesh, viscosity)
    e.solve()
    print(e.failed_to_solve)
    out = fd.File("temp_PDEConstrained_u.pvd")
    out.write(e.solution.split()[0])