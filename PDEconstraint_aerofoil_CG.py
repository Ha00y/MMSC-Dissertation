import firedrake as fd
from fireshape import PdeConstraint
import numpy as np
import netgen
from netgen.occ import *

from DG_mass_inv import DGMassInv

class NavierStokesSolverCG(PdeConstraint):
    """Incompressible Navier-Stokes as PDE constraint."""

    def __init__(self, mesh_m, Re, gamma):
        super().__init__()
        self.mesh_m = mesh_m
        self.failed_to_solve = False  # when self.solver.solve() fail
        self.gamma = gamma

        # Setup problem, Taylor-Hood finite elements
        self.V = fd.VectorFunctionSpace(self.mesh_m, "CG", 4) \
            * fd.FunctionSpace(self.mesh_m, "DG", 3)

        # Preallocate solution variables for state equation
        self.solution = fd.Function(self.V, name="State")
        self.testfunction = fd.TestFunction(self.V)

        # Define Reynolds' number
        self.Re = Re

        n = fd.FacetNormal(self.mesh_m)
        (x, y) = fd.SpatialCoordinate(self.mesh_m)
        p0 = 10/13 - x/13 #1atleft,0atright
        #f = fd.Constant((0,-9.81))
    
        # Weak form of incompressible Navier-Stokes equations
        z = self.solution
        u, p = fd.split(z)
        test = self.testfunction
        v, q = fd.split(test)

        # Define Lagrangian
        L = (
        0.5 * fd.inner(2/self.Re * fd.sym(fd.grad(u)), fd.sym(fd.grad(u)))*fd.dx
            + fd.inner(fd.dot(u,fd.grad(u)),u)*fd.dx
            -       fd.inner(p, fd.div(u))*fd.dx
            +       p0 * fd.inner(n, u)*fd.ds
            #-       fd.inner(f,u)*fd.dx
            + 0.5 * self.gamma * fd.inner(fd.div(u), fd.div(u))*fd.dx
            )

        # Optimality conditions
        self.F = fd.derivative(L, z)

        # Dirichlet Boundary conditions
        self.bcs = fd.DirichletBC(self.V.sub(0), fd.Constant((0,0)), (1,4,5))

        # PDE-solver parameters
        self.nsp = None
        #self.sp = {
        #            'snes_monitor': None,
        #            'snes_converged_reason': None,
        #            'snes_max_it': 20,
        #            'snes_atol': 1e-8,
        #            'snes_rtol': 1e-12,
        #            'snes_stol': 1e-06,
        #            'ksp_type': 'preonly',
        #            'pc_type': 'lu',
        #           'pc_factor_mat_solver_type': 'mumps'
        #            }
        
        self.sp = {
                    'mat_type': 'nest',
                    'snes_monitor': None,
                    'snes_converged_reason': None,
                    'snes_max_it': 20,
                    'snes_atol': 1e-8,
                    'snes_rtol': 1e-12,
                    'snes_stol': 1e-06,
                    'ksp_type': 'fgmres',
                    'ksp_converged_reason': None,
                    'ksp_monitor_true_residual': None,
                    'ksp_max_it': 500,
                    'ksp_atol': 1e-08,
                    'ksp_rtol': 1e-10,
                    'pc_type': 'fieldsplit',
                    'pc_fieldsplit_type': 'schur',
                    'pc_fieldsplit_schur_factorization_type': 'full',

                    'fieldsplit_0': {'ksp_convergence_test': 'skip',
                                    'ksp_max_it': 1,
                                    'ksp_norm_type': 'unpreconditioned',
                                    'ksp_richardson_self_scale': False,
                                    'ksp_type': 'richardson',
                                    'pc_type': 'mg',
                                        'pc_mg_type': 'full',
                                        'mg_coarse_assembled_pc_type': 'lu',
                                        'mg_coarse_assembled_pc_factor_mat_solver_type': 'mumps',
                                        'mg_coarse_pc_python_type': 'firedrake.AssembledPC',
                                        'mg_coarse_pc_type': 'python',
                                        'mg_levels': {'ksp_convergence_test': 'skip',
                                                    'ksp_max_it': 5,
                                                    'ksp_type': 'fgmres',
                                                    'pc_python_type': 'firedrake.ASMStarPC',
                                                    'pc_type': 'python'},
                                                    },

                    'fieldsplit_1': {'ksp_type': 'preonly',
                                    'pc_python_type': __name__ + '.DGMassInv',
                                    'pc_type': 'python'},
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
    Re = fd.Constant(1)
    gamma = fd.Constant(10000)

    N_x = 1000
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

    e = NavierStokesSolverCG(mesh, Re, gamma)
    e.solve()
    out = fd.File("temp_sol/temp_u.pvd")
    out.write(e.solution.subfunctions[0])