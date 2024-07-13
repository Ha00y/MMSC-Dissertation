from firedrake import *
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
        self.gamma = Constant(10000)
        self.sigma = Constant(10*(2+1)**2)
        self.theta = 1
        self.nu = viscosity

        # Setup problem
        self.V = FunctionSpace(self.mesh_m, "BDM", 2)  # Individual
        self.Q = FunctionSpace(self.mesh_m, "DG", 1)
        self.W = MixedFunctionSpace([self.V, self.Q])  # Mixed

        # Preallocate solution variables for state equation
        self.solution = Function(self.W, name="State")
        self.testfunction = TestFunction(self.W)

        n = FacetNormal(self.mesh_m)
        h = CellDiameter(self.mesh_m)
        (x, y) = SpatialCoordinate(self.mesh_m)
        p0 = 10/13 - x/13 #1atleft,0atright
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

        self.F = (
                   self.nu * a(u,v)
                 + c(u,v)
                 + b(v,p)
                 + b(u,q)
                 + self.nu * p0 * inner(n,v) * ds
                 )

        # Dirichlet Boundary conditions
        dirichlet_bids = (1, 4, 5)
        self.bcs = DirichletBC(self.W.sub(0), g_D, dirichlet_bids)
        for bid in dirichlet_bids:
            self.F += self.nu * a_bc(u, v, bid, g_D) + c_bc(u, v, bid, g_D)

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
            solve(self.F == 0, self.solution, bcs=self.bcs,
                     solver_parameters=self.sp)
        except ConvergenceError:
            self.failed_to_solve = True
            self.solution.assign(u_old)


if __name__ == "__main__":

    t = 0.12 # specify NACA00xx type
    viscosity = Constant(0.1)

    N_x = 1000
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
    ngsolve_mesh = Mesh(ngmesh)

    mh = MeshHierarchy(ngsolve_mesh, 2)
    mesh = mh[-1]

    e = NavierStokesSolver(mesh, viscosity)
    e.solve()
    out = File("temp_sol/temp_u.pvd")
    out.write(e.solution.subfunctions[0])
