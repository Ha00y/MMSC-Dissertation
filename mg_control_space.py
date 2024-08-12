import firedrake as fd
import fireshape as fs

class MultiGridControlSpace(fs.ControlSpace):
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
    """

    def __init__(self, mesh, refinements=1, degree=1):

        # one refinement level with `refinements`-many uniform refinements
        self.mh = fd.MeshHierarchy(mesh, refinements)

        # Coordinate function spaces on all meshes
        self.Vs = [fd.VectorFunctionSpace(mesh, "CG", degree) for mesh in self.mh]
        self.V_duals = [V.dual() for V in self.Vs]

        # Intermediate functions for prolongation and restriction
        self.tmp_funs = [fd.Function(V) for V in self.Vs]
        self.tmp_cofuns = [fd.Cofunction(V_dual) for V_dual in self.V_duals]  # not sure about this

        # Control space on most refined mesh
        self.mesh_r = self.mh[-1]
        self.V_r = self.Vs[-1]
        self.V_r_dual = self.V_duals[-1]
        element = self.V_r.ufl_element()
        self.ids = [fd.Function(mesh.coordinates) for mesh in self.mh]

        # Create self.id and self.T on most refined mesh
        X = fd.SpatialCoordinate(self.mesh_r)
        self.id = fd.Function(self.Vs[-1]).interpolate(X)
        self.T = fd.Function(self.Vs[-1], name="T")
        self.T.assign(self.id)
        self.mesh_m = fd.Mesh(self.T)
        self.V_m = fd.FunctionSpace(self.mesh_m, element)
        self.V_m_dual = self.V_m.dual()

    def restrict(self, residual, out):

        for (prev, next) in zip([residual] + self.tmp_cofuns[:-1][::-1], self.tmp_cofuns[:-1][::-1]):
            fd.restrict(prev, next)
   
        out.cofun.assign(self.tmp_cofuns[0])
        #print("restricted!", flush=True)

    def interpolate(self, vector, out):

        for (prev, next) in zip([vector.fun] + self.tmp_funs[1:], self.tmp_funs[1:]):
            fd.prolong(prev, next)

        out.assign(self.tmp_funs[-1])
        #print("prolonged!", flush=True)

    def update_domain(self, q: 'ControlVector'):

        out = fs.ControlSpace.update_domain(self, q)

        for (prev, next) in zip(self.tmp_funs[1:], self.tmp_funs[:-1]):
            fd.inject(prev, next)

        for (mesh, id, qh) in zip(self.mh, self.ids, self.tmp_funs):
            mesh.coordinates.assign(id + qh)

        for i, mesh in enumerate(self.mh):
            with fd.CheckpointFile(f'mesh_gen/naca0012_mesh_mg_{i}.h5', 'w') as afile:
                mesh.name = 'naca0012_mg'
                afile.save_mesh(mesh)

        return out

    def get_zero_vec(self):
        fun = fd.Function(self.Vs[0])
        return fun

    def get_zero_covec(self):
        fun = fd.Cofunction(self.V_duals[0])
        return fun

    def get_space_for_inner(self):
        return (self.Vs[0], None)

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