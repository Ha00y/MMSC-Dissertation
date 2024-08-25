import firedrake as fd
import firedrake.pyplot as fplt
from firedrake.mg.utils import get_level
import fireshape as fs
import fireshape.zoo as fsz

import ROL
import numpy as np
from ROL.numpy_vector import NumpyVector
import matplotlib.pyplot as plt

import netgen.meshing as ngm
import netgen
from netgen.occ import *

from PDEconstraint_cylinder_DG import NavierStokesSolverDG
from PDEconstraint_cylinder_CG import NavierStokesSolverCG

from CR_Hdot_inner_product import CRHdotInnerProduct
from objective_cylinder import CylinderObjective

# setup problem
disk = WorkPlane(Axes((3,0,0), n=Z, h=X)).Circle(1).Face()
rect = WorkPlane(Axes((-3,-3,0), n=Z, h=X)).Rectangle(12, 6).Face()
domain = rect - disk

geo = OCCGeometry(domain, dim=2)
ngmesh = geo.GenerateMesh(maxh=0.3)
mesh_m = fd.Mesh(ngmesh)

#fplt.triplot(mesh_m)
#plt.gca().legend()
#plt.show() 

Q = fs.FeControlSpace(mesh_m)
#Q = MultiGridControlSpace(mesh_m, degree=1, refinements=1)
#Q = fs.FeMultiGridControlSpace(mesh, refinements=2)

#inner = fs.LaplaceInnerProduct(Q, fixed_bids=[1, 2, 3, 4])
inner = CRHdotInnerProduct(Q, fixed_bids=[1, 2, 3, 4])
q = fs.ControlVector(Q, inner)

# setup PDE constraint
Re = 1
Fr = np.nan # Specify np.nan for no forcing term
gamma = 10000 

#e = NavierStokesSolverDG(Q.mesh_m, Re, Fr, gamma)
e = NavierStokesSolverCG(Q.mesh_m, Re, Fr, gamma)
e.solution.subfunctions[0].rename("Velocity")
e.solution.subfunctions[1].rename("Pressure")

out = fd.VTKFile("output/solution.pvd")

objective_values = []

def cb():

    objective_values.append(J.value(q, None))

    return out.write(e.solution.subfunctions[0])

# create PDEconstrained objective functional
J_ = CylinderObjective(e, Q, cb=cb)
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
                    'Iteration Limit': 100}}

params = ROL.ParameterList(params_dict, "Parameters")
problem = ROL.OptimizationProblem(J, q, econ=econ, emul=emul)
solver = ROL.OptimizationSolver(problem, params)
solver.solve()

print(objective_values)