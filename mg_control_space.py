import firedrake as fd
import fireshape as fs
from firedrake.mg.utils import get_level
from itertools import islice

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

        # Create another copy of mesh, to be updated with the T on the levels
        # This is hideous
        with fd.CheckpointFile("/tmp/argh.h5", "w") as f:
            mesh.name = "mesh"
            f.save_mesh(mesh)

        with fd.CheckpointFile("/tmp/argh.h5", "r") as f:
            cmesh_T = f.load_mesh("mesh")

        # mesh_T has lost all hierarchy information, restore
        self.mh_T = fd.MeshHierarchy(cmesh_T, refinements)
        self.Ts = [fd.Function(V, name="T") for V in self.Vs]

        for (mesh_T, T) in zip(self.mh_T, self.Ts):
            T.interpolate(fd.SpatialCoordinate(mesh_T))
            mesh_T.coordinates = T.topological

        self.T = self.Ts[-1]
        self.id = fd.Function(self.T)
        self.mesh_m = self.mh_T[-1]
        self.V_m = fd.FunctionSpace(self.mesh_m, element)
        self.V_m_dual = self.V_m.dual()

    def restrict(self, residual, out):

        self.tmp_cofuns[-1].dat.data[:] = residual.dat.data
        for (prev, next) in zip(self.tmp_cofuns[::-1], self.tmp_cofuns[:-1][::-1]):
            fd.restrict(prev, next)

        out.cofun.assign(self.tmp_cofuns[0])
        #print("restricted!", flush=True)

    def interpolate(self, vector, out):

        self.tmp_funs[0].dat.data[:] = vector.fun.dat.data
        for (prev, next) in zip(self.tmp_funs, self.tmp_funs[1:]):
            fd.prolong(prev, next)

        out.assign(self.tmp_funs[-1])
        #print("prolonged!", flush=True)

    def update_domain(self, q: 'ControlVector'):

        out = fs.ControlSpace.update_domain(self, q)

        (mh, _) = get_level(self.V_m.mesh())

        print("mh: ", mh, flush=True)

        i = 0
        fd.VTKFile(f"/tmp/coordinates-{i}.pvd").write(self.V_m.mesh().coordinates)

        for (prev, next) in zip(reversed(mh), reversed(mh[:-1])):
            import ipdb; ipdb.set_trace()
            fd.inject(prev.coordinates, next.coordinates)
            i += 1
            fd.VTKFile(f"/tmp/coordinates-{i}.pvd").write(next.coordinates)

        print("Updated domain", flush=True)
        import sys; sys.exit(1)

        #for i, mesh in enumerate(self.mh):
        #    with fd.CheckpointFile(f'mesh_gen/naca0012_mesh_mg_{i}.h5', 'w') as afile:
        #        mesh.name = 'naca0012_mg'
        #        afile.save_mesh(mesh)

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
