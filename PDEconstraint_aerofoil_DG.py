from firedrake import *
from fireshape import PdeConstraint

from DG_mass_inv import DGMassInv
from mesh_gen.naca_gen import NACAgen

class NavierStokesSolverDG(PdeConstraint):
    """Incompressible Navier-Stokes as PDE constraint."""

    def __init__(self, mesh_m, Re, gamma):
        super().__init__()
        self.mesh_m = mesh_m
        self.failed_to_solve = False  # when self.solver.solve() fail
        self.gamma = gamma
        self.sigma = Constant(10*(2+1)**2)
        self.theta = 1
        self.Re = Re

        # Setup problem
        self.V = FunctionSpace(self.mesh_m, "BDM", 2)  # Individual
        self.Q = FunctionSpace(self.mesh_m, "DG", 1)
        self.W = MixedFunctionSpace([self.V, self.Q])  # Mixed

        # Preallocate solution variables for state equation
        self.solution = Function(self.W, name="State")
        self.testfunction = TestFunction(self.W)

        n = FacetNormal(self.mesh_m)
        h = CellDiameter(self.mesh_m)
        (x, y) = SpatialCoordinate(self.mesh_m)
        p0 = 10/13 - x/13 #1atleft,0atright
        g_D = Constant((0,0))

        # Weak form of incompressible Navier-Stokes equations
        z = self.solution
        u, p = split(z)
        test = self.testfunction
        v, q = split(test)

        # Define Lagrangian
        def a(u, v):
            return inner(2*sym(grad(u)), grad(v))*dx \
                 - inner(avg(2*sym(grad(u))), 2*avg(outer(v, n))) * dS \
                 - inner(avg(2*sym(grad(v))), 2*avg(outer(u, n))) * dS \
                 + self.sigma/avg(h) * inner(2*avg(outer(u,n)),2*avg(outer(v,n))) * dS

        def a_bc(u, v, bid, g):
            return -inner(outer(v,n),2*sym(grad(u)))*ds(bid) \
                   -inner(outer(u-g,n),2*sym(grad(v)))*ds(bid) \
                   +(self.sigma/h)*inner(v,u-g)*ds(bid)

        def c(u, v):
            uflux_int = 0.5*(dot(u, n) + abs(dot(u, n)))*u
            return - dot(u ,div(outer(v,u)))*dx \
                   + dot(v('+')-v('-'), uflux_int('+')-uflux_int('-'))*dS

        def c_bc(u, v, bid, g):
            if g is None:
                uflux_ext = 0.5*(inner(u,n)+abs(inner(u,n)))*u
            else:
                uflux_ext = 0.5*(inner(u,n)+abs(inner(u,n)))*u + 0.5*(inner(u,n)-abs(inner(u,n)))*g
            return dot(v,uflux_ext)*ds(bid)

        def b(u, q):
            return div(u) * q * dx
        
        L_lagrangian = 0.5 * self.gamma * inner(div(u), div(u)) * dx
        F_lagrangian = derivative(L_lagrangian, z)

        self.F = (
                   (1/self.Re) * a(u,v)
                 + c(u,v)
                 + b(v,p)
                 + b(u,q)
                 + (1/self.Re) * p0 * inner(n,v) * ds
                 + F_lagrangian
                 )

        # Dirichlet Boundary conditions
        dirichlet_bids = (1, 4, 5)
        self.bcs = DirichletBC(self.W.sub(0), g_D, dirichlet_bids)
        for bid in dirichlet_bids:
            self.F += (1/self.Re) * a_bc(u, v, bid, g_D) + c_bc(u, v, bid, g_D)

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
        #            'pc_factor_mat_solver_type': 'mumps'
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
                    'pc_fieldsplit_schur_factorization_type': 'upper',

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
            solve(self.F == 0, self.solution, bcs=self.bcs,
                     solver_parameters=self.sp)
        except ConvergenceError:
            self.failed_to_solve = True
            self.solution.assign(u_old)


if __name__ == "__main__":

    Re = Constant(1)
    gamma = Constant(10000)
    profile = '0012' # specify NACA type
    
    mesh = NACAgen(profile)

    # Load the mesh
    #with CheckpointFile('naca0012_mesh.h5', 'r') as afile:
    #    mesh = afile.load_mesh('naca0012')

    e = NavierStokesSolverDG(mesh, Re, gamma)
    e.solve()
    out = File("temp_sol/temp_u.pvd")
    out.write(e.solution.subfunctions[0])