from .innerproduct import InnerProduct
import ROL
import firedrake as fd

# new imports for splines
from firedrake.petsc import PETSc
from functools import reduce
from scipy.interpolate import splev
import numpy as np

class FeMultiGridControlSpace(ControlSpace):
    """
    FEControlSpace on given mesh and StateSpace on uniformly refined mesh.

    Use the provided mesh to construct a Lagrangian finite element control
    space. Then, create a finer mesh using `refinements`-many uniform
    refinements and construct a representative of ControlVector that is
    compatible with the state space.

    Inputs:
        refinements: type int, number of uniform refinements to perform
                     to obtain the StateSpace mesh.
        degree: type int, degree of Lagrange basis functions of ControlSpace.

    Note: as of 15.11.2023, higher-order meshes are not supported, that is,
    mesh_r has to be a polygonal mesh (mesh_m can still be of higher-order).
    """

    def __init__(self, mesh_r, refinements=1, degree=1):

        # one refinement level with `refinements`-many uniform refinements
        self.mh = fd.MeshHierarchy(mesh_r, refinements)

        # Control space on all but most refined mesh
        self.V_r_coarse = [fd.VectorFunctionSpace(self.mh[i], "CG", degree) for i in range(refinements)]
        self.V_r_coarse_dual = [V_r.dual() for V_r in self.V_r_coarse]

        # Control space on most refined mesh
        self.mesh_r = self.mh[-1]
        element = self.V_r_coarse[-1].ufl_element()
        self.V_r = fd.FunctionSpace(self.mesh_r, element)
        self.V_r_dual = self.V_r.dual()

        # Create self.id and self.T on most refined mesh
        X = fd.SpatialCoordinate(self.mesh_r)
        self.id = fd.Function(self.V_r).interpolate(X)
        self.T = fd.Function(self.V_r, name="T")
        self.T.assign(self.id)
        self.mesh_m = fd.Mesh(self.T)
        self.V_m = fd.FunctionSpace(self.mesh_m, element)
        self.V_m_dual = self.V_m.dual()

    def restrict(self, residual, out):
        
        current_residual = residual
        next_residual = self.V_r_coarse_dual[-1]
        for i in range(len(self.V_r_coarse_dual)-1):
            fd.restrict(current_residual, next_residual.cofun)
            current_residual = next_residual  # Update current residual to the restricted one
            next_residual = self.V_r_coarse_dual[-i-2]  # Update next residual to the next coarser one

    #def restrict(self, residual, out):
    #    fd.restrict(residual, out.cofun)
    #    restrict from fine to coarse

    def interpolate(self, vector, out):
        current_vector = vector
        next_vector = self.V_r_coarse[1]
        for i in range(len(self.V_r_coarse)-1):
            fd.prolong(current_vector, next_vector)
            current_vector = next_vector  # Update current vector to the prolonged one
            next_vector = self.V_r_coarse[i+1]  # Update next vector to the next coarser one
    
    # def interpolate(self, vector, out):
    #     fd.prolong(vector.fun, out)
    #    prolong from coarse to fine

    def get_zero_vec(self):
        fun = fd.Function(self.V_r)
        fun *= 0.
        return fun

    def get_zero_covec(self):
        fun = fd.Cofunction(self.V_r_dual)
        fun *= 0.
        return fun

    def get_space_for_inner(self):
        return (self.V_r, None)

    def store(self, vec, filename="control"):
        """
        Store the vector to a file to be reused in a later computation.
        DumbCheckpoint requires that the mesh, FunctionSpace and parallel
        decomposition are identical between store and load.

        """
        with fd.DumbCheckpoint(filename, mode=fd.FILE_CREATE) as chk:
            chk.store(vec.fun, name=filename)

    def load(self, vec, filename="control"):
        """
        Load a vector from a file.
        DumbCheckpoint requires that the mesh, FunctionSpace and parallel
        decomposition are identical between store and load.
        """
        with fd.DumbCheckpoint(filename, mode=fd.FILE_READ) as chk:
            chk.load(vec.fun, name=filename)