import firedrake as fd
from fireshape import UflInnerProduct
import numpy as np

class CRHdotInnerProduct(UflInnerProduct):

    def get_weak_form(self, V):

        self.alpha = fd.Constant(1e-2)
        self.CR = True # Choose H^1 or CR inner product
        self.sym = True # Choose H^1 or H(sym) inner product
        self.clamp = False # Choose use of mu

        u = fd.TrialFunction(V)
        v = fd.TestFunction(V)

        Bu = self.cauchy_riemann(u); Bv = self.cauchy_riemann(v)
        Du = self.derivative(u); Dv = self.derivative(v)

        if self.clamp:
            d = self.compute_distance_function(V.mesh())
            epsilon = fd.Constant(0.01)
            mu = fd.sqrt(epsilon / (d + epsilon))

        if self.CR:
            if self.sym:
                Eu = 0.5 * (Du + Du.T); Ev = 0.5 * (Dv + Dv.T)
                if self.clamp:
                    return (1/self.alpha) * fd.inner(mu*Bu, mu*Bv) * fd.dx + fd.inner(Eu, Ev) * fd.dx
                else:
                    return (1/self.alpha) * fd.inner(Bu, Bv) * fd.dx + fd.inner(Eu, Ev) * fd.dx
            else:
                if self.clamp:
                    return (1/self.alpha) * fd.inner(mu*Bu, mu*Bv) * fd.dx + fd.inner(Du, Dv) * fd.dx
                else:
                    return (1/self.alpha) * fd.inner(Bu, Bv) * fd.dx + fd.inner(Du, Dv) * fd.dx
        else:
            return fd.inner(Du, Dv) * fd.dx


    def cauchy_riemann(self, u):
        return fd.as_vector([fd.Dx(u[0], 0) - fd.Dx(u[1], 1), fd.Dx(u[1], 0) + fd.Dx(u[0], 1)])

    def derivative(self,u):
        return fd.as_matrix([[fd.Dx(u[0], 0), fd.Dx(u[0], 1)], [fd.Dx(u[1], 0), fd.Dx(u[1], 1)]])
    
    def compute_distance_function(self, mesh):
        # Function space for the distance function
        V = fd.FunctionSpace(mesh, "CG", 1)
        d = fd.Function(V)

        # Solve an auxiliary PDE to approximate the distance to the boundary
        # The PDE: (grad d)·(grad v) = 1 in the interior and d = 0 on the boundary
        v = fd.TestFunction(V)
        boundary_conditions = fd.DirichletBC(V, 0, "on_boundary")
        F = fd.dot(fd.grad(d), fd.grad(v)) * fd.dx - v * fd.Constant(1) * fd.dx
        fd.solve(F == 0, d, bcs=boundary_conditions)
       
        #fd.VTKFile("temp_sol/distance.pvd").write(d)
        return d

#   def get_nullspace(self, V):
#        """This nullspace contains constant functions."""
#        dim = V.value_size
#        if dim == 2:
#            n1 = fd.Function(V).interpolate(fd.Constant((1.0, 0.0)))
#            n2 = fd.Function(V).interpolate(fd.Constant((0.0, 1.0)))
#            res = [n1, n2]
#        elif dim == 3:
#            n1 = fd.Function(V).interpolate(fd.Constant((1.0, 0.0, 0.0)))
#            n2 = fd.Function(V).interpolate(fd.Constant((0.0, 1.0, 0.0)))
#            n3 = fd.Function(V).interpolate(fd.Constant((0.0, 0.0, 1.0)))
#            res = [n1, n2, n3]
#        else:
#            raise NotImplementedError
#        return res
