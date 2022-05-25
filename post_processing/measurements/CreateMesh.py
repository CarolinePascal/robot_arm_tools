import numpy as np
import pyvista as pv
import gmsh as gmsh
import meshio as meshio
import sys as sys

import matplotlib.pyplot as plt

#Load points from .csv file
points0 = np.loadtxt('measurements/17062021_S-/Positions.csv',dtype=np.float,delimiter=',')[:,:3]
points1 = np.loadtxt('measurements/17062021_S0/Positions.csv',dtype=np.float,delimiter=',')[:,:3]
points2 = np.loadtxt('measurements/17062021_S+/Positions.csv',dtype=np.float,delimiter=',')[:,:3]

#points0 -= (points0.max(axis=0) + points0.min(axis=0))/2
#points1 -= (points1.max(axis=0) + points1.min(axis=0))/2
#points2 -= (points2.max(axis=0) + points2.min(axis=0))/2

points = [points0,points1,points2]

#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
#ax.scatter(points0[:,0],points0[:,1],points0[:,2])
#plt.show()

gmsh.initialize(sys.argv)

#Create a surfacic mesh for each sphere using 3D Delaunay triangulation, and save it as .ply 
for i,set in enumerate(points):
    cloud = pv.PolyData(set[:82])
    volume = cloud.delaunay_3d(alpha=2.)
    shell = volume.extract_geometry()
    shell.save("measurements/Mesh"+str(i)+".ply")

#Create a volumic mesh corresponding to the most inner spheres, and save it as .msh
gmsh.merge("measurements/Mesh0.ply")
gmsh.merge("measurements/Mesh1.ply")

n = gmsh.model.getDimension()
s = gmsh.model.getEntities(n)

l = gmsh.model.geo.addSurfaceLoop([s[i][1] for i in range(len(s))])
gmsh.model.geo.addVolume([l])

gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(3)
gmsh.write("measurements/innerMesh.mesh")
gmsh.clear()

#Create a volumic mesh corresponding to the most outer spheres, and save it as .msh
gmsh.merge("measurements/Mesh1.ply")
gmsh.merge("measurements/Mesh2.ply")

n = gmsh.model.getDimension()
s = gmsh.model.getEntities(n)

l = gmsh.model.geo.addSurfaceLoop([s[i][1] for i in range(len(s))])
gmsh.model.geo.addVolume([l])

gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(3)
gmsh.write("measurements/outerMesh.mesh")
gmsh.clear()

#Merge the two half volumic meshes, and save it as a 2.2 version .msh (=> FreeFem compatibility)
gmsh.merge("measurements/innerMesh.mesh")
gmsh.merge("measurements/outerMesh.mesh")
gmsh.model.mesh.removeDuplicateNodes()
gmsh.model.mesh.generate(3)
gmsh.write("measurements/measurementsMesh.mesh")

#mesh = meshio.read("measurementsMesh.msh")
#mesh.write("measurementsMesh.msh", file_format="gmsh22", binary=False)s

#Display 
if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()