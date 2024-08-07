import firedrake as fd
from firedrake.mg.utils import get_level
import fireshape as fs
import fireshape.zoo as fsz
import ROL
import numpy as np

from PDEconstraint_aerofoil_CG import NavierStokesSolverCG
from PDEconstraint_aerofoil_DG import NavierStokesSolverDG

from CR_Hdot_inner_product import CRHdotInnerProduct
from objective_aerofoil import AerofoilObjective
from cauchy_riemann import CauchyRiemannConstraint

# setup problem
with fd.CheckpointFile('mesh_gen/naca0012_mesh.h5', 'r') as afile:
        mesh = afile.load_mesh('naca0012')
mh = fd.MeshHierarchy(mesh, 2)
mesh_m = mh[0]

#Q = fs.FeControlSpace(mesh_m)
Q = fs.FeMultiGridControlSpace(mesh_m, degree=2, refinements=2)
#inner = fs.LaplaceInnerProduct(Q, fixed_bids=[1, 2, 3, 4])
inner = CRHdotInnerProduct(Q, fixed_bids=[1, 2, 3, 4])
q = fs.ControlVector(Q, inner)

# setup PDE constraint
#nu = 1.506*(10**-5)
#L = 0.3; u = 10
#Re = fd.Constant(u*L/nu)	
#Fr = fd.Constant(u/(9.81*L)**0.5)
Re = 1 # CHANGE these in DGMassInv too
Fr = np.nan # Specify np.nan for no forcing term
gamma = 10000 # CHANGE these in DGMassInv too

e = NavierStokesSolverDG(Q.mesh_m, Re, Fr, gamma)
#e = NavierStokesSolverCG(Q.mesh_m, Re, Fr, gamma)
e.solution.subfunctions[0].rename("Velocity")
e.solution.subfunctions[1].rename("Pressure")

# save state variable evolution in file u2.pvd or u3.pvd
if mesh_m.topological_dimension() == 2:  # in 2D
    out = fd.File("output/solution.pvd")
elif mesh_m.topological_dimension() == 3:  # in 3D
    out = fd.File("output/solution3D.pvd")

def cb():

    (mh, level) = get_level(e.solution.subfunctions[0].function_space().mesh())
    mh[0].name = 'naca0012_shapeopt'
    with fd.CheckpointFile('mesh_gen/naca0012_mesh_shapeopt.h5', 'w') as afile:
        afile.save_mesh(mh[0])

    #for i, mesh in enumerate(mh):
    #    mesh.name = f'naca0012_{i}'
    #    with fd.CheckpointFile(f'mesh_gen/naca0012_mesh_shapeopt_{i}.h5', 'w') as afile:
    #        afile.save_mesh(mesh)

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