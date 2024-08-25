import numpy as np
import firedrake as fd
import matplotlib.pyplot as plt
import firedrake.pyplot as fplt

from LidCavitysolver_DG import NavierStokesSolverDG

Ns = [1, 2, 4, 8, 16]
Re = 1

norm_l2 = np.zeros(len(Ns))
h = np.zeros(len(Ns))

for i, N in enumerate(Ns):

    mesh = fd.UnitSquareMesh(N, N)#, diagonal="crossed")
    #mesh = fd.RectangleMesh(N, N, 2, 2, diagonal="crossed")
    mh = fd.MeshHierarchy(mesh, 1)
    mesh_m = mh[-1]

    h[i] = mesh_m.cell_sizes.dat.data.max()

    (x, y) = fd.SpatialCoordinate(mesh_m)

    ux_true = 8*(x**4 - 2*x**3 + x**2)*(4*y**3 - 2*y)
    uy_true = -8*(4*x**3 - 6*x**2 + 2*x)*(y**4 - y**2)
    
    #ux_true = 0.25*(x - 2)**2 * x**2 * y * (y**2 - 2)
    #uy_true = -0.25*x*(x**2 - 3*x + 2)* y**2 * (y**2 - 4)

    u_true = fd.as_vector([ux_true, uy_true])

    e = NavierStokesSolverDG(mesh_m, Re)
    e.solve()

    (u, p) = e.solution.subfunctions

    out = fd.VTKFile("temp_sol/temp_u.pvd")
    out.write(e.solution.subfunctions[0])

    norm_l2[i] = fd.norm(fd.grad(u - u_true), 'L2')

print(norm_l2)
plt.figure()
plt.loglog(h, norm_l2, '-o')
plt.loglog(h, h, '--', label='$\mathcal{O}(h)$',color='grey',)
plt.legend(loc = 4, fontsize='9'); plt.xlabel('$h$'); plt.ylabel('$\| \nabla (\mathbf{u} - \mathbf{u}_h) \|_{L^2(\Omega)}$'); plt.show()

