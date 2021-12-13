import numpy as np
from numpy.__config__ import show
import pyvista as pv

N = 100
Rmin = 0.2
Rmax = 0.3

pointsInt = np.empty((N,3))
points = np.empty((2*N,3))

for i in range(N):
    points[2*i,:] = np.array([Rmin*np.cos(i*np.pi/N),Rmin*np.sin(i*np.pi/N),0]) 
    points[2*i+1,:] = np.array([Rmax*np.cos(i*np.pi/N),Rmax*np.sin(i*np.pi/N),0]) 
    pointsInt[i,:] = np.array([Rmin*np.cos(i*np.pi/N),Rmin*np.sin(i*np.pi/N),0]) 

cloudInt = pv.PolyData(pointsInt)
surfaceInt = cloudInt.delaunay_2d(alpha=1.0)

cloud = pv.PolyData(points)
surface = cloud.delaunay_2d(alpha=1.0,edge_source=surfaceInt)
surface.plot()





