import firedrake as fd
import fireshape as fs
from firedrake.mg.utils import get_level
from itertools import islice

class MultiGridControlSpace(fs.ControlSpace):
    def __init__(self, mesh, refinements=1, degree=1):

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

        # Make a bunch of vector transformations
        self.Ts = [fd.Function(V, name="T") for V in self.Vs]
        [T.interpolate(fd.SpatialCoordinate(m)) for (m, T) in zip(self.mh, self.Ts)]

        # Expose public variables
        self.T = self.Ts[-1]
        self.id = fd.Function(self.T)

        # Make mapped meshes
        mapped_meshes = [fd.Mesh(T) for T in self.Ts]

        # Make new mesh hierarchy out of this
        self.mh_mapped = fd.HierarchyBase(mapped_meshes, self.mh.coarse_to_fine_cells,
                                                         self.mh.fine_to_coarse_cells,
                                                         self.mh.refinements_per_level,
                                                         self.mh.nested)

        # Expose more public variables
        self.mesh_m = self.mh_mapped[-1]
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

        i = 0
        fd.VTKFile(f"/tmp/coordinates-{i}.pvd").write(self.V_m.mesh().coordinates)

        Ts = self.Ts
        for (prev, next) in zip(Ts[::-1], Ts[::-1][1:]):
            fd.inject(prev, next)
            i += 1
            fd.VTKFile(f"/tmp/coordinates-{i}.pvd").write(next)

        print("Updated domain", flush=True)
        #import sys; sys.exit(1)

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
