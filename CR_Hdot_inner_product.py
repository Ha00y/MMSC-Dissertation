import firedrake as fd
from fireshape import UflInnerProduct
import numpy as np

class CRHdotInnerProduct(UflInnerProduct):

    def get_weak_form(self, V):

        self.alpha = fd.Constant(0.02)
        self.sym = True # Choose H^1 or H(sym) inner product

        u = fd.TrialFunction(V)
        v = fd.TestFunction(V)

        Bu = self.cauchy_riemann(u); Bv = self.cauchy_riemann(v)
        Du = self.derivative(u); Dv = self.derivative(v)

        d = self.find_boundary(V.mesh())
        epsilon = fd.Constant(0.01)
        mu = fd.sqrt(epsilon / (d + epsilon))


        if self.sym:
            Eu = 0.5 * (Du + Du.T); Ev = 0.5 * (Dv + Dv.T)
            return (1/self.alpha) * fd.inner(Bu, Bv) * fd.dx + fd.inner(Eu, Ev) * fd.dx
        else:
            return (1/self.alpha) * fd.inner(mu*Bu, mu*Bv) * fd.dx + fd.inner(Du, Dv) * fd.dx
    
    def cauchy_riemann(self, u):
        return fd.as_vector([fd.Dx(u[0], 0) - fd.Dx(u[1], 1), fd.Dx(u[1], 0) + fd.Dx(u[0], 1)])

    def derivative(self,u):
        return fd.as_matrix([[fd.Dx(u[0], 0), fd.Dx(u[0], 1)], [fd.Dx(u[1], 0), fd.Dx(u[1], 1)]])
    
    def find_boundary(self, mesh):

        V = fd.FunctionSpace(mesh, "CG", 1)
        all_node_coords = mesh.coordinates.dat.data
        boundary_nodes = fd.DirichletBC(V, 0, "on_boundary").nodes
        boundary_node_coords = all_node_coords[boundary_nodes]

        distances = np.zeros(all_node_coords.shape[0])

        for i, coord in enumerate(all_node_coords):
            distances[i] = np.min(np.linalg.norm(boundary_node_coords - coord, axis=1))

        distance_function = fd.Function(V)
        distance_function.dat.data[:] = distances

        return distance_function


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
