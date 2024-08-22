from firedrake import *
from fireshape import PdeConstraint
import numpy as np

class MyTransferManager(object):
    def __init__(self):
        self.builtin_tm = TransferManager(use_averaging=False)

    def prolong(self, source, target):
        print("Inside my prolongation!", flush=True)
        self.builtin_tm.prolong(source, target)
        div_s = norm(div(source))
        div_t = norm(div(target))
        print("||div(source)||: ", div_s)
        print("||div(target)||: ", div_t)

        if div_t >= 2*div_s:
            print("Warning. Prolongation not continuous?")
            V_s = source.function_space()
            Q_s = FunctionSpace(V_s.mesh(), "DG", 1)
            div_source = project(div(source), Q_s)
            div_source.rename("DivergenceSource")
            V_t = target.function_space()
            Q_t = FunctionSpace(V_t.mesh(), "DG", 1)
            div_target = project(div(target), Q_t)
            div_target.rename("DivergenceTarget")
            File("/tmp/source.pvd").write(source, div_source)
            File("/tmp/target.pvd").write(target, div_target)
            print("Saving PVD files and quitting.")
            import sys; sys.exit(1)


    def restrict(self, source, target):
        self.builtin_tm.restrict(source, target)

    def inject(self, source, target):
        self.builtin_tm.inject(source, target)


class NavierStokesSolverDG(PdeConstraint):
    """Incompressible Navier-Stokes as PDE constraint."""

    def __init__(self, mesh, Re, Fr, gamma):
        super().__init__()
        self.failed_to_solve = False  # when self.solver.solve() fail
        self.mesh = mesh
        self.gamma = Constant(gamma)
        self.sigma = Constant(10*(2+1)**2)
        self.Re = Constant(Re)
        self.Fr = Constant(Fr)
        
        f = Constant((0,-1))
        g_D = Constant((0,0))

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
        p0 = 10/13 - x/13 #1atleft,0atright

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
        #lagrangian = 0.5 * self.gamma * inner(project(div(u),self.Q), div(v)) * dx
    
        self.F = (
                   (1/self.Re) * a(u,v)
                 + c(u,v)
                 + b(v,p)
                 + b(u,q)
                 + p0 * inner(n,v) * ds
                 + F_lagrangian
                # + lagrangian
                 )
        
        if np.isnan(Fr) == False:
            self.F -= (1/self.Fr)**2 * inner(f,v) * dx
        
        # Dirichlet Boundary conditions
        dirichlet_bids = (1, 4, 5)
        self.bcs = DirichletBC(self.W.sub(0), g_D, dirichlet_bids)
        for bid in dirichlet_bids:
            self.F += (1/self.Re) * a_bc(u, v, bid, g_D) + c_bc(u, v, bid, g_D)

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
                    #'mg_transfer_manager': __name__ + ".MyTransferManager",

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
                                     'ksp_richardson_scale': -(2/float(Re) + float(gamma)),
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
    Fr = np.nan # Specify np.nan for no forcing term
    gamma = 10000

    # Load the mesh
    with CheckpointFile('mesh_gen/mesh_0.h5', 'r') as afile:
    #with CheckpointFile('mesh_gen/naca0012_mesh.h5', 'r') as afile:
        mesh_import = afile.load_mesh('naca0012')
    #with CheckpointFile(f'mesh_gen/meshT_0.h5', 'w') as afile:
    #    mesh_m = afile.load_mesh('mesh')
    
    VTKFile(f"mesh_gen/mesh_initial.pvd").write(mesh_import.coordinates)
    mh = MeshHierarchy(mesh_import, 2)
    mesh_new = mh[-1]
    for i in range(3):
        VTKFile(f"mesh_gen/mesh_after_{i}.pvd").write(mh[i].coordinates)

    e = NavierStokesSolverDG(mesh_new, Re, Fr, gamma)
    e.solve()
    out = VTKFile("temp_sol/temp_u.pvd")
    out.write(e.solution.subfunctions[0])
