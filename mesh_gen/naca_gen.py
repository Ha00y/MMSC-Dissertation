import firedrake as fd
import netgen
from netgen.occ import *

from math import cos, sin, tan
from math import atan
from math import pi
from math import pow
from math import sqrt

def NACAgen(profile):
    
    n = 1000
    xNACA = naca4(profile, n, False, False)[0]
    yNACA = naca4(profile, n, False, False)[1]
    pnts = [Pnt(xNACA[i], yNACA[i], 0) for i in range(len(xNACA))]

    spline = SplineApproximation(pnts)
    aerofoil = Face(Wire(spline)).Move((0.3,1,0))
    rect = WorkPlane(Axes((-1, 0, 0), n=Z, h=X)).Rectangle(4, 2).Face()
    domain = rect - aerofoil

    domain.edges.name="wing"
    domain.edges.Min(Y).name="bottom"
    domain.edges.Max(Y).name="top"
    domain.edges.Min(X).name="inlet"
    domain.edges.Max(X).name="outlet"
    geo = OCCGeometry(domain, dim=2)

    ngmesh = geo.GenerateMesh(maxh=1)
    mesh = fd.Mesh(ngmesh)
    mesh.name = 'naca0012'
    #mh = fd.MeshHierarchy(mesh, 2)
    #mesh = mh[-1]

    return mesh

def naca4(number, n, finite_TE = False, half_cosine_spacing = False):
    """
    Returns 2*n+1 points in [0 1] for the given 4 digit NACA number string
    """

    m = float(number[0])/100.0
    p = float(number[1])/10.0
    t = float(number[2:])/100.0

    a0 = +0.2969
    a1 = -0.1260
    a2 = -0.3516
    a3 = +0.2843

    if finite_TE:
        a4 = -0.1015 # For finite thick TE
    else:
        a4 = -0.1036 # For zero thick TE

    if half_cosine_spacing:
        beta = linspace(0.0,pi,n+1)
        x = [(0.5*(1.0-cos(xx))) for xx in beta]  # Half cosine based spacing
    else:
        x = linspace(0.0,1.0,n+1)

    yt = [5*t*(a0*sqrt(xx)+a1*xx+a2*pow(xx,2)+a3*pow(xx,3)+a4*pow(xx,4)) for xx in x]

    xc1 = [xx for xx in x if xx <= p]
    xc2 = [xx for xx in x if xx > p]

    if p == 0:
        xu = x
        yu = yt

        xl = x
        yl = [-xx for xx in yt]

        xc = xc1 + xc2
        zc = [0]*len(xc)
    else:
        yc1 = [m/pow(p,2)*xx*(2*p-xx) for xx in xc1]
        yc2 = [m/pow(1-p,2)*(1-2*p+xx)*(1-xx) for xx in xc2]
        zc = yc1 + yc2

        dyc1_dx = [m/pow(p,2)*(2*p-2*xx) for xx in xc1]
        dyc2_dx = [m/pow(1-p,2)*(2*p-2*xx) for xx in xc2]
        dyc_dx = dyc1_dx + dyc2_dx

        theta = [atan(xx) for xx in dyc_dx]

        xu = [xx - yy * sin(zz) for xx,yy,zz in zip(x,yt,theta)]
        yu = [xx + yy * cos(zz) for xx,yy,zz in zip(zc,yt,theta)]

        xl = [xx + yy * sin(zz) for xx,yy,zz in zip(x,yt,theta)]
        yl = [xx - yy * cos(zz) for xx,yy,zz in zip(zc,yt,theta)]

    X = xu[::-1] + xl[1:]
    Z = yu[::-1] + yl[1:]

    return X,Z

def linspace(start,stop,np):
    """
    Emulate Matlab linspace
    """
    return [start+(stop-start)*i/(np-1) for i in range(np)]


if __name__ == "__main__":

    # Generate the mesh
    mesh = NACAgen('0012')

    # Save the mesh
    with fd.CheckpointFile('mesh_gen/naca0012_mesh.h5', 'w') as afile:
        afile.save_mesh(mesh)

    # Load the mesh
    #with fd.CheckpointFile('naca0012_mesh.h5', 'r') as afile:
    #    loaded_mesh = afile.load_mesh('naca0012')

    # Compare coordinates
    #for i in range(len(mesh.coordinates.dat.data)):
    #    if (mesh.coordinates.dat.data[:][i,0] == loaded_mesh.coordinates.dat.data[:][i,0]) == False:
    #        print('fail')
    #    if (mesh.coordinates.dat.data[:][i,1] == loaded_mesh.coordinates.dat.data[:][i,1]) == False:
    #        print('fail')

    #for marker in mesh.exterior_facets.unique_markers:
    #    print(f"Marker: {marker}")
    #for marker in loaded_mesh.exterior_facets.unique_markers:
    #    print(f"Marker: {marker}")