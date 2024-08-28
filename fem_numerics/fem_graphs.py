import numpy as np
import firedrake as fd
import matplotlib.pyplot as plt
import firedrake.pyplot as fplt

from LidCavitysolver_DG import NavierStokesSolverDG
from LidCavitysolver_CG import NavierStokesSolverCG

Ns = [2, 4, 8, 16, 32]
Re = [1, 10, 100]

u_norm_l2 = np.zeros((2,len(Re),len(Ns)))
#p_norm_l2 = np.zeros(len(Ns))
h = np.zeros(len(Ns))

for j in range(len(Re)):
    for i, N in enumerate(Ns):

        mesh = fd.UnitSquareMesh(N, N)
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

        f = x**4 - 2*x**3 + x**2
        g = y**4 - y**2

        F = 0.2*x**5 - 0.5*x**4 + (1/3)*x**3
        F1 = -4*x**6 + 12*x**5 - 14*x**4 + 8*x**3 - 2*x**2
        F2 = 0.5 *(x**4 - 2*x**3 + x**2)**2
        G1 = -24*y**5 + 8*y**3 - 4*y

        #p_true = (8 / Re[j]) * (F*(24*y) + (4*x**3 - 6*x**2 + 2*x)*(4*y**3 - 2*y)) + 64*F2*(g*(12*y**2 - 2) - (4*y**3 - 2*y)**2)

        print("DG, Re = ", Re[j], "N = ", N)
        eDG = NavierStokesSolverDG(mesh_m, Re[j])
        eDG.solve()

        (uDG, pDG) = eDG.solution.subfunctions

        u_norm_l2[1, j, i] = fd.norm(fd.grad(uDG - u_true), 'L2')

        print("CG, Re = ", Re[j], "N = ", N)
        eCG = NavierStokesSolverCG(mesh_m, Re[j])
        eCG.solve()

        (uCG, pCG) = eCG.solution.subfunctions

        u_norm_l2[0, j, i] = fd.norm(fd.grad(uCG - u_true), 'L2')
        #p_norm_l2[i] = fd.norm(p - p_true, 'L2')

        #out = fd.VTKFile("temp_sol/temp_u.pvd")
        #out.write(e.solution.subfunctions[0])


plt.figure()
for j in range(len(Re)):
    plt.loglog(h, u_norm_l2[0,j,:], '-o', label=f'T-H, Re = {Re[j]}')
for j in range(len(Re)):
    plt.loglog(h, u_norm_l2[1,j,:], '-o', label=f'$H$(div), Re = {Re[j]}')
#plt.loglog(h, p_norm_l2, '-o')
plt.loglog(h, h**2, '--', label='$\mathcal{O}(h^2)$',color='grey',)
plt.legend(loc = 4, fontsize='9'); plt.xlabel('$h$'); plt.ylabel('$\| \nabla (\mathbf{u} - \mathbf{u}_h) \|_{L^2(\Omega)}$'); plt.show()

print('h:',h)
print('u:', u_norm_l2)

