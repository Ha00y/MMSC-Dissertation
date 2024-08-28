import firedrake as fd
from firedrake.mg.utils import get_level
import fireshape as fs
import fireshape.zoo as fsz
import ROL
import numpy as np
from ROL.numpy_vector import NumpyVector

from PDEconstraint_aerofoil_CG import NavierStokesSolverCG
from PDEconstraint_aerofoil_DG import NavierStokesSolverDG

from CR_Hdot_inner_product import CRHdotInnerProduct
from biharm_inner_product import BiharmInnerProduct
from drag_objective import DragObjective
from lift_objective import LiftObjective
from mg_control_space import MultiGridControlSpace
from naca_gen import NACAgen

# setup problem
mesh_m = NACAgen('2412')
#mesh_m = NACAgen('0012')

#Q = fs.FeControlSpace(mesh_m)
Q = MultiGridControlSpace(mesh_m, degree=1, refinements=1)

inner = CRHdotInnerProduct(Q, fixed_bids=[1, 2, 3, 4])
#inner = BiharmInnerProduct(Q, fixed_bids=[1, 2, 3, 4])
q = fs.ControlVector(Q, inner)

# setup PDE constraint
#nu = 1.506*(10**-5)
#L = 0.3; u = 10
#Re = fd.Constant(u*L/nu)	
#Fr = fd.Constant(u/(9.81*L)**0.5)
Re = 100
Fr = np.nan # Specify np.nan for no forcing term
gamma = 10000 

e = NavierStokesSolverDG(Q.mesh_m, Re, Fr, gamma)
#e = NavierStokesSolverCG(Q.mesh_m, Re, Fr, gamma)
e.solution.subfunctions[0].rename("Velocity")
e.solution.subfunctions[1].rename("Pressure")

out = fd.VTKFile("output/solution.pvd")

def cb():
    return out.write(e.solution.subfunctions[0])

# create PDEconstrained objective functional
J_ = LiftObjective(e, Q, cb=cb)
#J_ = DragObjective(e, Q, cb=cb)
J = fs.ReducedObjective(J_, e)

# add regularization to improve mesh quality
J_q = fsz.MoYoSpectralConstraint(10, fd.Constant(0.5), Q)

J = J + J_q 

# Set up constraints
vol = fsz.VolumeFunctional(Q)
initial_vol = vol.value(q, None)
econ = fs.EqualityConstraint([vol], target_value=[initial_vol])
emul = NumpyVector(1)

# ROL parameters
params_dict = {
    'General': {
        'Print Verbosity': 0,  # Set to 1 to understand output
        'Secant': {
            'Type': 'Limited-Memory BFGS',
            'Maximum Storage': 10
        }
    },
    'Step': {
        'Type': 'Augmented Lagrangian',
        'Augmented Lagrangian': {
            'Subproblem Step Type': 'Trust Region',
            'Trust Region': {
                'Initial Radius': 1,  # Control initial step size (default is 1.0)
                'Minimum Radius': 1e-6,
                'Maximum Radius': 3,
                'Eta': 0.1,  # For scaling the radius
                'Gamma': 0.5  # Step size reduction factor
            },
            'Print Intermediate Optimization History': False,
            'Subproblem Iteration Limit': 10
        }
    },
    'Status Test': {
        'Gradient Tolerance': 1e-4,
        'Step Tolerance': 1e-4,
        'Constraint Tolerance': 1e-3,
        'Iteration Limit': 10
    }
}


params = ROL.ParameterList(params_dict, "Parameters")
problem = ROL.OptimizationProblem(J, q, econ=econ, emul=emul)
solver = ROL.OptimizationSolver(problem, params)
solver.solve()
