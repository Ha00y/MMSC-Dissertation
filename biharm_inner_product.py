import firedrake as fd
from fireshape import UflInnerProduct
import numpy as np

class BiharmInnerProduct(UflInnerProduct):

    def get_weak_form(self, V):

        self.alpha = fd.Constant(1e-6)
        self.beta = fd.Constant(1e-3)
        self.sigma = fd.Constant(10)   

        u = fd.TrialFunction(V)
        v = fd.TestFunction(V)
        n = fd.FacetNormal(V.mesh())
        h_E = fd.CellDiameter(V.mesh())

        Bu = self.cauchy_riemann(u); Bv = self.cauchy_riemann(v)
        Du = self.derivative(u); Dv = self.derivative(v)

        Eu = 0.5 * (Du + Du.T); Ev = 0.5 * (Dv + Dv.T)

        lap_u = fd.div(fd.grad(u))
        lap_v = fd.div(fd.grad(v))
                
        grad_u_n = fd.dot(fd.grad(u), n)
        grad_v_n = fd.dot(fd.grad(v), n)
                
        d2u_dn2 = self.second_derivative_normal(u, n)
        d2v_dn2 = self.second_derivative_normal(v, n)
        
        a = fd.inner(lap_u, lap_v) * fd.dx - fd.inner(fd.avg(lap_u), fd.jump(grad_v_n)) * fd.dS - fd.inner(fd.avg(lap_v), fd.jump(grad_u_n)) * fd.dS + self.sigma / fd.avg(h_E) * fd.inner(fd.jump(d2u_dn2), fd.jump(d2v_dn2)) * fd.dS
  
        return (1/self.alpha) * fd.inner(Bu, Bv) * fd.dx + fd.inner(Eu, Ev) * fd.dx + self.beta * a


    def cauchy_riemann(self, u):
        return fd.as_vector([fd.Dx(u[0], 0) - fd.Dx(u[1], 1), fd.Dx(u[1], 0) + fd.Dx(u[0], 1)])


    def derivative(self,u):
        return fd.as_matrix([[fd.Dx(u[0], 0), fd.Dx(u[0], 1)], [fd.Dx(u[1], 0), fd.Dx(u[1], 1)]])
    

    def second_derivative_normal(self, u, n):
        hessian_u = fd.grad(fd.grad(u))  # Hessian of u
        return fd.dot(n, fd.dot(hessian_u, n))  # nᵀ H(u) n (second derivative in the normal direction)


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
