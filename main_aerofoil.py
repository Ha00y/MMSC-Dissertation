import firedrake as fd
import fireshape as fs
import fireshape.zoo as fsz
import numpy as np
import ROL
import netgen
from netgen.occ import *

from PDEconstraint_aerofoil_CG import NavierStokesSolverCG
from PDEconstraint_aerofoil_DG import NavierStokesSolverDG

from CR_Hdot_inner_product import CRHdotInnerProduct

from objective_aerofoil import AerofoilObjective
from cauchy_riemann import CauchyRiemannConstraint

# setup problem
t = 0.12 # specify NACA00xx type

N_x = 1000
x = np.linspace(0,1.0089,N_x) 

def naca00xx(x,t):
  y = 5*t*(0.2969*(x**0.5) - 0.1260*x - 0.3516*(x**2) + 0.2843*(x**3) - 0.1015*(x**4))
  return np.concatenate((x,np.flip(x)),axis=None), np.concatenate((y,np.flip(-y)),axis=None)

x, y = naca00xx(x,t)

pnts = [Pnt(x[i], y[i], 0) for i in range(len(x))]

spline = SplineApproximation(pnts)
#aerofoil = Face(Wire(spline)).Move((0.3,1,0)).Rotate(Axis((0.3,1,0), Z), -10)
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