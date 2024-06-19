# %%
import matplotlib.pyplot as plt
import numpy as np

import netgen
import ROL
from firedrake import *
from fireshape import *
from netgen.occ import *

import fireshape.zoo as fsz

# %%
t = 0.12 # specify NACA00xx type

N_x = 100
x = np.linspace(0,1.0089,N_x)

def naca00xx(x,t):
  y = 5*t*(0.2969*(x**0.5) - 0.1260*x - 0.3516*(x**2) + 0.2843*(x**3) - 0.1015*(x**4))
  return np.concatenate((x,np.flip(x)),axis=None), np.concatenate((y,np.flip(-y)),axis=None)

x, y = naca00xx(x,t)

pnts = [Pnt(x[i], y[i], 0) for i in range(len(x))]

spline = SplineApproximation(pnts)
aerofoil = Face(Wire(spline)).Move((0.3,0.5,0)).Rotate(Axis((0.3,0.5,0), Z), -10)
rect = WorkPlane(Axes((-1, 0, 0), n=Z, h=X)).Rectangle(4, 1).Face()
domain = rect - aerofoil

domain.edges.name="wing"
domain.edges.Min(Y).name="bottom"
domain.edges.Max(Y).name="top"
domain.edges.Min(X).name="inlet"
domain.edges.Max(X).name="outlet"
geo = OCCGeometry(domain, dim=2)

ngmesh = geo.GenerateMesh(maxh=1)
ngsolve_mesh = Mesh(ngmesh)

mh = MeshHierarchy(ngsolve_mesh, 2)
mesh = mh[-1]

# %%
class DGMassInv(PCBase):
  def initialize(self, pc):
    _, P = pc.getOperators()
    appctx = self.get_appctx(pc)
    V = dmhooks.get_function_space(pc.getDM())

    # get function spaces
    u = TrialFunction(V)
    v = TestFunction(V)
    massinv = assemble(Tensor(inner(u, v)*dx).inv)
    self.massinv = massinv.petscmat

  def update(self, pc):
    pass

  def apply(self, pc, x, y):
    self.massinv.mult(x, y)
    scaling = 1/float(Re) + float(gamma)
    y.scale(-scaling)

  def applyTranspose(self, pc, x, y):
    raise NotImplementedError("Sorry!")

# %%
class solve_navier_stokes(PdeConstraint):
    """Incompressible Navier-Stokes as PDE constraint."""

    def __init__(self, mesh_init, Re, gamma, sp):
        super().__init__()
        self.mesh_init = mesh_init
        self.failed_to_solve = False  # when self.solver.solve() fail
        self.sp = sp
        self.Re = Re
        self.gamma = gamma

        n = FacetNormal(self.mesh_init)
        (x, y) = SpatialCoordinate(self.mesh_init)

        # Define Scott--Vogelius function space W
        self.V = VectorFunctionSpace(self.mesh_init, "CG", 4)
        self.Q = FunctionSpace(self.mesh_init, "DG", 3)
        self.W = MixedFunctionSpace([self.V, self.Q])

        self.bcs = DirichletBC(self.W.sub(0), Constant((0,0)), (1,4,5))

        self.w = Function(self.W, name="Solution")
        (self.u, self.p) = split(self.w)
        (v, q) = split(TestFunction(self.W))

        p0 = 10/13 - x/13 #1atleft,0atright
        #f = Constant((0,-9.81))

        # Define Lagrangian
        L = (
        0.5 * inner(2/self.Re * sym(grad(self.u)), sym(grad(self.u)))*dx
            + inner(dot(self.u,grad(self.u)),self.u)*dx
            -       inner(self.p, div(self.u))*dx
            +       p0 * inner(n, self.u)*ds
            #-       inner(f,self.u)*dx
            + 0.5 * self.gamma * inner(div(self.u), div(self.u))*dx
            )

        # Optimality conditions
        self.F = derivative(L, self.w)

    def solve(self):
        super().solve()
        self.failed_to_solve = False

        w_old = self.w.copy(deepcopy=True)

        try:
            solve(self.F == 0, self.w, self.bcs, solver_parameters=self.sp) # Monitor incompressibility
        except ConvergenceError:
            self.failed_to_solve = True
            self.w.assign(w_old)

        #(u_, p_) = self.w.subfunctions
        #u_.rename("Velocity")
        #p_.rename("Pressure")

        #return u_, p_

# %%
sp = {
'mat_type': 'nest',
'snes_monitor': None,
'snes_converged_reason': None,
'snes_max_it': 20,
'snes_atol': 1e-8,
'snes_rtol': 1e-12,
'snes_stol': 1e-06,
'ksp_type': 'fgmres',
'ksp_converged_reason': None, 'ksp_monitor_true_residual': None,
'ksp_max_it': 500,
'ksp_atol': 1e-08,
'ksp_rtol': 1e-10,
'pc_type': 'fieldsplit',
'pc_fieldsplit_type': 'schur', 'pc_fieldsplit_schur_factorization_type': 'full',

'fieldsplit_0': {'ksp_convergence_test': 'skip',
                 'ksp_max_it': 1,
'ksp_norm_type': 'unpreconditioned', 'ksp_richardson_self_scale': False, 'ksp_type': 'richardson',
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

sp = {
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

# %%
class Objective(ShapeObjective):
    """L2 tracking functional for Poisson problem."""

    def __init__(self, pde_solver: solve_navier_stokes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pde_solver = pde_solver

    def value_form(self):
        """Evaluate misfit functional."""

        if self.pde_solver.failed_to_solve:  # return NaNs if state solve fails
            return np.nan * dx(self.pde_solver.mesh_init)
        else:
            w = self.pde_solver.w
            u, p = split(w)
            return inner(grad(u), grad(u)) * dx

# %%
# setup problem
Q = FeControlSpace(mesh)
inner_Q = LaplaceInnerProduct(Q, fixed_bids=[1, 2, 3, 4])
q = ControlVector(Q, inner_Q)

# Define Reynolds number and gamma
Re = Constant(10)
gamma = Constant(10000)

# setup PDE constraint
e = solve_navier_stokes(Q.mesh_m, Re, gamma, sp)
e.w.subfunctions[0].rename("Velocity")
e.w.subfunctions[1].rename("Pressure")

out = VTKFile("output/solution.pvd")
def cb(*args):
    out.write(*e.w.subfunctions)

# create PDEconstrained objective functional
J_ = Objective(e, Q, cb=cb)
J = ReducedObjective(J_, e)

# add regularization to improve mesh quality
Jq = fsz.MoYoSpectralConstraint(10, Constant(0.5), Q)
J = J + Jq

# ROL parameters
params_dict = {
    'General': {'Print Verbosity': 0,  # set to 1 to understand output
                'Secant': {'Type': 'Limited-Memory BFGS',
                           'Maximum Storage': 10}},
    'Step': {'Type': 'Augmented Lagrangian',
             'Augmented Lagrangian':
             {'Subproblem Step Type': 'Trust Region',
              'Print Intermediate Optimization History': False,
              'Subproblem Iteration Limit': 10}},
    'Status Test': {'Gradient Tolerance': 1e-2,
                    'Step Tolerance': 1e-2,
                    'Constraint Tolerance': 1e-1,
                    'Iteration Limit': 10}}
params = ROL.ParameterList(params_dict, "Parameters")
problem = ROL.OptimizationProblem(J, q)
solver = ROL.OptimizationSolver(problem, params)
solver.solve()

# %%



