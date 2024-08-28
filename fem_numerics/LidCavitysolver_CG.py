import firedrake as fd
from fireshape import PdeConstraint
import numpy as np

class NavierStokesSolverCG(PdeConstraint):
    """Incompressible Navier-Stokes as PDE constraint."""

    def __init__(self, mesh, Re):
        super().__init__()
        self.failed_to_solve = False  # when self.solver.solve() fail
        self.mesh_m = mesh
        self.gamma = fd.Constant(10000)
        self.Re = fd.Constant(Re)

        # Setup problem, Taylor-Hood finite elements
        k = 2
        self.V = fd.VectorFunctionSpace(self.mesh_m, "CG", k) \
            * fd.FunctionSpace(self.mesh_m, "CG", k-1, variant="integral")

        # Preallocate solution variables for state equation
        self.solution = fd.Function(self.V, name="State")
        self.testfunction = fd.TestFunction(self.V)

        n = fd.FacetNormal(self.mesh_m)
        (x, y) = fd.SpatialCoordinate(self.mesh_m)

        f = x**4 - 2*x**3 + x**2
        g = y**4 - y**2

        F = 0.2*x**5 - 0.5*x**4 + (1/3)*x**3
        F1 = -4*x**6 + 12*x**5 - 14*x**4 + 8*x**3 - 2*x**2
        F2 = 0.5 * f**2
        G1 = -24*y**5 + 8*y**3 - 4*y

        B = (-8 / self.Re) * (24*F + 2*(4*x**3 - 6*x**2 + 2*x)*(12*y**2 - 2) + (24*x - 12)*g) - 64*(F2*G1 - g*(4*y**3 -2*y)*F1)
        
        u_lid = 16*(x**4 -2*x**3 + x**2) 
        #u_lid = (x**2)*(2 - x)**2

        g_lid = fd.as_vector((u_lid, 0))
        g_D = fd.Constant((0,0))
    
        # Weak form of incompressible Navier-Stokes equations
        z = self.solution
        u, p = fd.split(z)
        test = self.testfunction
        v, q = fd.split(test)

        L_lagrangian = 0.5 * self.gamma * fd.inner(fd.div(u), fd.div(u)) * fd.dx
        F_lagrangian = fd.derivative(L_lagrangian, z)

        self.F = (
            fd.inner(2/self.Re*fd.sym(fd.grad(u)), fd.grad(v)) * fd.dx
            - fd.div(u)*q * fd.dx
            - fd.div(v)*p * fd.dx
            + fd.inner(u, fd.grad(u)*v)* fd.dx
            #- fd.inner(fd.outer(u, u), fd.sym(fd.grad(v))) * fd.dx
            #+ fd.inner(fd.dot(fd.outer(u, u), n), v) * fd.ds
            + F_lagrangian
            - fd.inner(fd.as_vector((0,-B)),v) * fd.dx
            )

        # Boundary conditions
        zero_bids = (1, 2, 3)
        self.bcs = [fd.DirichletBC(self.V.sub(0), g_D, zero_bids), fd.DirichletBC(self.V.sub(0), g_lid, 4)]

        # PDE-solver parameters
        self.nsp = None
        self.sp_lu = {
                    'snes_monitor': None,
                    'snes_converged_reason': None,
                    'snes_max_it': 20,
                    'snes_atol': 1e-6,
                    'snes_rtol': 1e-12,
                    'snes_stol': 1e-5,
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

                    'fieldsplit_1': {'ksp_type': 'richardson',
                                     'ksp_max_it': 1,
                                     'ksp_convergence_test': 'skip',
                                     'ksp_richardson_scale': -(2/float(self.Re) + float(self.gamma)),
                                     'pc_type': 'python',
                                     'pc_python_type': 'firedrake.MassInvPC',
                                     'Mp_pc_type': 'jacobi'},
                    }


    def solve(self):
        super().solve()
        self.failed_to_solve = False
        u_old = self.solution.copy(deepcopy=True)
        try:
            fd.solve(self.F == 0, self.solution, bcs=self.bcs,
                     solver_parameters=self.sp_mg)
        except fd.ConvergenceError:
            self.failed_to_solve = True
            self.solution.assign(u_old)


if __name__ == "__main__":

    Re = 1
    Fr = np.nan # Specify np.nan for no forcing term
    gamma = 10000
   
    # Load the mesh
    with fd.CheckpointFile('mesh_gen/naca0012_mesh.h5', 'r') as afile:
        mesh = afile.load_mesh('naca0012')

    e = NavierStokesSolverCG(mesh, Re, Fr, gamma)
    e.solve()
    out = fd.VTKFile("temp_sol/temp_u.pvd")
    out.write(e.solution.subfunctions[0])