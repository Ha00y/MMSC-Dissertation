from firedrake import *
from fireshape import PdeConstraint
import numpy as np

class NavierStokesSolverDG(PdeConstraint):
    """Incompressible Navier-Stokes as PDE constraint."""

    def __init__(self, mesh, Re):
        super().__init__()
        self.failed_to_solve = False  # when self.solver.solve() fail
        self.mesh = mesh
        self.gamma = Constant(10000)
        self.sigma = Constant(10*(2+1)**2)
        self.Re = Constant(Re)

        # Setup problem
        k = 2
        self.V = FunctionSpace(self.mesh, "BDM", k)  # Individual
        self.Q = FunctionSpace(self.mesh, "DG", k-1, variant="integral")
        self.W = MixedFunctionSpace([self.V, self.Q])  # Mixed

        # Preallocate solution variables for state equation
        self.solution = Function(self.W, name="State")
        self.testfunction = TestFunction(self.W)

        n = FacetNormal(self.mesh)
        h = CellDiameter(self.mesh)
        (x, y) = SpatialCoordinate(self.mesh)

        f = x**4 - 2*x**3 + x**2
        g = y**4 - y**2

        F = 0.2*x**5 - 0.5*x**4 + (1/3)*x**3
        F1 = -4*x**6 + 12*x**5 - 14*x**4 + 8*x**3 - 2*x**2
        F2 = 0.5 *(x**4 - 2*x**3 + x**2)**2
        G1 = -24*y**5 + 8*y**3 - 4*y

        B = -8 / self.Re * (24 * F + 2 * Dx(f, 0) * Dx(Dx(g, 1),1) + Dx(Dx(Dx(f, 0),0),0) * g
                    - 64 * (F2 * G1 - g * Dx(g, 1) * F1))
        
        u_lid = 16*(x**4 -2*x**3 + x**2) 
        #u_lid = (x**2)*(2 - x)**2

        g_lid = as_vector((u_lid, 0))
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
                 + F_lagrangian
                 - inner(as_vector((0,-B)),v) * dx
                 )
        
        # Boundary conditions
        zero_bids = (1, 2, 3)
        self.bcs = [DirichletBC(self.W.sub(0), g_D, zero_bids), DirichletBC(self.W.sub(0), g_lid, 4)]

        g = [g_D, g_D, g_D, g_lid]
        for bid in range(1, 5):
            self.F += (1/self.Re) * a_bc(u, v, bid, g[bid-1]) + c_bc(u, v, bid, g[bid-1])
    

        # PDE-solver parameters
        self.nsp = None
        self.sp_lu = {
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

        self.sp_mg = {
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
                                        'mg_coarse_pc_type': 'lu',
                                        'mg_coarse_ksp_type': 'richardson',
                                        'mg_coarse_pc_factor_mat_solver_type': 'mumps',
                                        'mg_levels': {'ksp_convergence_test': 'skip',
                                                    #'ksp_monitor_true_residual': None,
                                                    'ksp_max_it': 5,
                                                    'ksp_type': 'fgmres',
                                                    'pc_python_type': 'firedrake.ASMStarPC',
                                                    'pc_type': 'python'},
                                                    },

                    'fieldsplit_1': {'ksp_type': 'richardson',
                                     'ksp_max_it': 1,
                                     'ksp_convergence_test': 'skip',
                                     'ksp_richardson_scale': -(2/float(Re) + float(self.gamma)),
                                     'pc_type': 'python',
                                     'pc_python_type': 'firedrake.MassInvPC',
                                     'Mp_pc_type': 'jacobi'},
                    }

    def solve(self):
        super().solve()
        self.failed_to_solve = False
        u_old = self.solution.copy(deepcopy=True)
        try:
            solve(self.F == 0, self.solution, bcs=self.bcs,
                     solver_parameters=self.sp_mg)
        except ConvergenceError:
            self.failed_to_solve = True
            self.solution.assign(u_old)


if __name__ == "__main__":

    Re = 1
    N = 10

    mesh = UnitSquareMesh(N, N, diagonal="crossed")
    mh = MeshHierarchy(mesh, 1)
    mesh_m = mh[-1]
    
    e = NavierStokesSolverDG(mesh_m, Re)
    e.solve()
    out = VTKFile("temp_sol/temp_u.pvd")
    out.write(e.solution.subfunctions[0])
