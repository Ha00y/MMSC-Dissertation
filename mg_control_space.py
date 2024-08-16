import firedrake as fd
import fireshape as fs

class MultiGridControlSpace(fs.ControlSpace):
    def __init__(self, mesh, refinements=1, degree=1):

        self.mh = fd.MeshHierarchy(mesh, refinements)

        # Coordinate function spaces on all meshes
        self.Vs = [fd.VectorFunctionSpace(mesh, "CG", degree) for mesh in self.mh]
        self.V_duals = [V.dual() for V in self.Vs]

        # Control space on most refined mesh
        self.mesh_r = self.mh[-1]
        self.V_r = self.Vs[-1]
        self.V_r_dual = self.V_duals[-1]

        # Create hierarchy of transformations, identities, and cofunctions
        self.ids = [fd.Function(V) for V in self.Vs]
        [id_.interpolate(fd.SpatialCoordinate(m)) for (m, id_) in zip(self.mh, self.ids)]
        self.Ts = [fd.Function(V, name="T") for V in self.Vs]
        [T.assign(id_) for (T, id_) in zip(self.Ts, self.ids)]
        self.cofuns = [fd.Cofunction(V_dual) for V_dual in self.V_duals]
        self.T = self.Ts[-1]

        # Create mesh hierarchy reflecting self.Ts
        mapped_meshes = [fd.Mesh(T) for T in self.Ts]
        self.mh_mapped = fd.HierarchyBase(mapped_meshes, self.mh.coarse_to_fine_cells,
                                                         self.mh.fine_to_coarse_cells,
                                                         self.mh.refinements_per_level,
                                                         self.mh.nested)

        # Create moved mesh and V_m (and dual) to evaluate and collect shape derivative
        self.mesh_m = self.mh_mapped[-1]
        element = self.Vs[-1].ufl_element()
        self.V_m = fd.FunctionSpace(self.mesh_m, element)
        self.V_m_dual = self.V_m.dual()

    def restrict(self, residual, out):

        self.cofuns[-1].assign(residual)
        for (prev, next) in zip(self.cofuns[::-1], self.cofuns[:-1][::-1]):
            fd.restrict(prev, next)
        out.cofun.assign(self.cofuns[0])

    def interpolate(self, vector, out):
        # out is unused, but keep it for API compatibility
        self.Ts[0].assign(vector.fun)
        for (prev, next) in zip(self.Ts, self.Ts[1:]):
            fd.prolong(prev, next)

    def update_domain(self, q: 'ControlVector'):  # not happy of overwriting this
        """
        Update the interpolant self.T with q
        """

        # Check if the new control is different from the last one.  ROL is
        # sometimes a bit strange in that it calls update on the same value
        # more than once, in that case we don't want to solve the PDE again.
        if not hasattr(self, 'lastq') or self.lastq is None:
            self.lastq = q.clone()
            self.lastq.set(q)
        else:
            self.lastq.axpy(-1., q)
            # calculate l2 norm (faster)
            diff = self.lastq.vec_ro().norm()
            self.lastq.axpy(+1., q)
            if diff < 1e-20:
                return False
            else:
                self.lastq.set(q)
        # pass slef.Ts only for API compatibility
        q.to_coordinatefield(self.Ts)
        # add identity to every function in the hierarchy
        # this is the only reason we need to overwrite update_domain
        #[T += id_ for (T, id_) in zip(self.Ts, self.ids)]
        [T.assign(T + id_) for (T, id_) in zip(self.Ts, self.ids)]
        return True

    # kept this as a comment for your debugging utils
    #def update_domain(self, q: 'ControlVector'):

    #    out = fs.ControlSpace.update_domain(self, q)

    #    i = 0
    #    fd.VTKFile(f"tmp/coordinates-{i}.pvd").write(self.V_m.mesh().coordinates)

    #    Ts = self.Ts
    #    for (prev, next) in zip(Ts[::-1], Ts[::-1][1:]):
    #        fd.inject(prev, next)
    #        i += 1
    #        fd.VTKFile(f"tmp/coordinates-{i}.pvd").write(next)

    #    for T in self.Ts:
    #        fd.VTKFile(f"tmp/Ts-{i}.pvd").write(T)


    #    #print("Updated domain", flush=True)
    #    #import sys; sys.exit(1)

    #    #for i, mesh in enumerate(self.mh):
    #    #    with fd.CheckpointFile(f'mesh_gen/naca0012_mesh_mg_{i}.h5', 'w') as afile:
    #    #        mesh.name = 'naca0012_mg'
    #    #        afile.save_mesh(mesh)

    #    return out

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
