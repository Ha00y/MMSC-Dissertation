import firedrake as fd
from fireshape import PdeConstraint
import numpy as np
import netgen
from netgen.occ import *

class NavierStokesSolver(PdeConstraint):
    """Incompressible Navier-Stokes as PDE constraint."""

    def __init__(self, mesh_m, viscosity):
        super().__init__()
        self.mesh_m = mesh_m
        self.failed_to_solve = False  # when self.solver.solve() fail
        self.gamma = fd.Constant(10000)
        self.sigma = fd.Constant(10*(2+1)**2)
        self.theta = 1
        self.nu = viscosity

        # Setup problem
        V = fd.FunctionSpace(mesh, "BDM", 2)  # Individual
        Q = fd.FunctionSpace(mesh, "DG", 1)
        self.W = fd.MixedFunctionSpace([V, Q])  # Mixed

        # Preallocate solution variables for state equation
        self.solution = fd.Function(self.W, name="State")
        self.testfunction = fd.TestFunction(self.W)

        n = fd.FacetNormal(self.mesh_m)
        h = fd.CellDiameter(self.mesh_m)
        (x, y) = fd.SpatialCoordinate(self.mesh_m)
        p0 = 10/13 - x/13 #1atleft,0atright
        #f = Constant((0,-9.81))
        g_D = fd.Constant((0,0))
    
        # Weak form of incompressible Navier-Stokes equations
        z = self.solution
        u, p = fd.split(z)
        test = self.testfunction
        v, q = fd.split(test)

        # Define Lagrangian
        def a_h(u,v):
            return (fd.inner(fd.grad(u), fd.grad(v)) * fd.dx
            + (self.sigma/h('+')) * fd.inner(fd.jump(u),fd.jump(v)) * fd.dS
            - fd.inner(fd.avg(fd.grad(u)) * n('+'), fd.jump(v)) * fd.dS
            - fd.inner(fd.jump(u), fd.avg(fd.grad(v)) * n('+')) * fd.dS
            )
        
        def a_h_partial(g_D,v):
            return ((self.sigma/h) * fd.inner(g_D,v) * fd.ds
            - fd.inner(g_D, fd.grad(v) * n) * fd.ds
            )
        
        def b_h(u,q):
            return (- q * fd.div(u) * fd.dx
            + fd.inner(fd.jump(u), n('+'))* fd.avg(q) * fd.dS
            )
        
        def b_h_partial(g_D,q):
            return fd.inner(g_D, n) * q * fd.ds
        
        def c_h(w,u,v):
            return (fd.inner(w, fd.grad(fd.inner(u,v))) * fd.dx 
            + 0.5 * fd.div(w) * fd.inner(u,v) * fd.dx 
            - fd.inner(fd.avg(w), n('+')) * fd.inner(fd.jump(u), fd.avg(v)) * fd.dS
            - 0.5 * fd.inner(fd.jump(w), n('+')) * fd.avg(fd.inner(u,v)) * fd.dS
            + (self.theta/2) * abs(fd.inner(fd.avg(w), n('+'))) * fd.inner(fd.jump(u), fd.jump(v)) * fd.dS
            )       

        def c_partial(g_D,u,v):
            return -0.5 * fd.inner(g_D,n) * fd.inner(u,v) * fd.ds

        self.F = (self.nu * a_h(u,v)) + c_h(u,u,v) + b_h(v,p) + b_h(u,q) - (self.nu * a_h_partial(g_D,v)) - c_partial(g_D,u,v) - b_h_partial(g_D,q) 


        # Dirichlet Boundary conditions
        self.bcs = fd.DirichletBC(self.W.sub(0), g_D, (1,4,5))

        # PDE-solver parameters
        self.nsp = None
        self.sp = {
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

    def solve(self):
        super().solve()
        self.failed_to_solve = False
        u_old = self.solution.copy(deepcopy=True)
        try:
            fd.solve(self.F == 0, self.solution, bcs=self.bcs,
                     solver_parameters=self.sp)
        except fd.ConvergenceError:
            self.failed_to_solve = True
            self.solution.assign(u_old)


if __name__ == "__main__":

    t = 0.12 # specify NACA00xx type
    viscosity = fd.Constant(0.1)

    N_x = 100
    x = np.linspace(0,1.0089,N_x)

    def naca00xx(x,t):
        y = 5*t*(0.2969*(x**0.5) - 0.1260*x - 0.3516*(x**2) + 0.2843*(x**3) - 0.1015*(x**4))
        return np.concatenate((x,np.flip(x)),axis=None), np.concatenate((y,np.flip(-y)),axis=None)

    x, y = naca00xx(x,t)

    pnts = [Pnt(x[i], y[i], 0) for i in range(len(x))]

    spline = SplineApproximation(pnts)
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

    e = NavierStokesSolver(mesh, viscosity)
    e.solve()
    out = fd.File("temp_sol/temp_u.pvd")
    out.write(e.solution.subfunctions[0])