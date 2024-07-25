import firedrake as fd
import fireshape as fs
import fireshape.zoo as fsz
import ROL

from PDEconstraint_aerofoil_CG import NavierStokesSolverCG
from PDEconstraint_aerofoil_DG import NavierStokesSolverDG

from CR_Hdot_inner_product import CRHdotInnerProduct
from objective_aerofoil import AerofoilObjective
from cauchy_riemann import CauchyRiemannConstraint

from mesh_gen.naca_gen import NACAgen

# setup problem
profile = '0012' # specify NACA type
mesh = NACAgen(profile)

Q = fs.FeControlSpace(mesh)
#inner = fs.LaplaceInnerProduct(Q, fixed_bids=[1, 2, 3, 4])
inner = CRHdotInnerProduct(Q, fixed_bids=[1, 2, 3, 4])
q = fs.ControlVector(Q, inner)

# setup PDE constraint
Re = fd.Constant(1) # CHANGE these in DGMassInv too
gamma = fd.Constant(10000) # CHANGE these in DGMassInv too

e = NavierStokesSolverDG(Q.mesh_m, Re, gamma)
#e = NavierStokesSolverCG(Q.mesh_m, Re, gamma)
e.solution.subfunctions[0].rename("Velocity")
e.solution.subfunctions[1].rename("Pressure")

# save state variable evolution in file u2.pvd or u3.pvd
if mesh.topological_dimension() == 2:  # in 2D
    out = fd.File("output/solution.pvd")
elif mesh.topological_dimension() == 3:  # in 3D
    out = fd.File("output/solution3D.pvd")

def cb():
    return out.write(e.solution.subfunctions[0])

# create PDEconstrained objective functional
J_ = AerofoilObjective(e, Q, cb=cb)
J = fs.ReducedObjective(J_, e)

# add regularization to improve mesh quality
J_q = fsz.MoYoSpectralConstraint(10, fd.Constant(0.5), Q)
J_cr = CauchyRiemannConstraint(e, Q, cb=cb)

J = J + J_q + J_cr

# Set up constraints
vol = fsz.VolumeFunctional(Q)
initial_vol = vol.value(q, None)
econ = fs.EqualityConstraint([vol], target_value=[initial_vol])
emul = ROL.StdVector(1)

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
    'Status Test': {'Gradient Tolerance': 1e-4,
                    'Step Tolerance': 1e-4,
                    'Constraint Tolerance': 1e-1,
                    'Iteration Limit': 10}}

params = ROL.ParameterList(params_dict, "Parameters")
problem = ROL.OptimizationProblem(J, q, econ=econ, emul=emul)
solver = ROL.OptimizationSolver(problem, params)
solver.solve()