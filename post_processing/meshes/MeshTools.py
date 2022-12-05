import numpy as np
import os

import subprocess
import anti_lib_progs as anti_lib_progs

from scipy.spatial import ConvexHull

import meshio

## Function creating a spheric mesh using an icosahedric approximation
#  @param radius Radius of the sphere
#  @param resolution Target resolution of the mesh
def GenerateSphericMesh(radius,resolution,save=False):

    #Icosahedron initial number of faces 
    k = 20

    #Tool functions
    T = lambda b,c:  b**2 + c**2 + b*c  #Triangulation number

    TTarget = int(np.round((16*np.pi*radius**2)/(k*np.sqrt(3)*resolution**2)))
    maxBound = int(np.ceil(np.sqrt(TTarget))) + 1

    solutions = np.empty((maxBound,maxBound))

    for i in range(maxBound):
        for j in range(i,maxBound):
            solutions[i,j] = T(i,j)
            solutions[j,i] = T(i,j)

    solution = np.unravel_index(np.argmin(np.abs(TTarget - solutions), axis=None), solutions.shape) 
    bTarget, cTarget = solution[0], solution[1]

    args = " -l -o /tmp/points.txt -p i -c " + str(bTarget) + "," + str(cTarget)
    subprocess.call("python3.8 " + os.path.dirname(anti_lib_progs.__file__) + "/geodesic.py " + args, shell=True)

    points = []
    with open("/tmp/points.txt",'r') as file:
        points = file.read().splitlines()

    points = np.array([np.array(line.split(" ")).astype(float) for line in points])
    points *= radius

    hull = ConvexHull(points)
    faces = hull.simplices

    if(save):
        meshPath = os.path.dirname(os.path.realpath(__file__)) + "/meshes/sphere/S_" + str(radius) + "_" + str(resolution) + ".mesh"
        print("Saving mesh at " + meshPath)
        meshio.write_points_cells(meshPath, list(points), [("triangle",list(faces))])

    return(points, faces)

if __name__ == "__main__":
    import sys

    radius = 0.1
    resolution = 0.01
    save = 0
    info = 0

    try:
        radius = float(sys.argv[1])
        resolution = float(sys.argv[2])
        save = int(sys.argv[3])
        info = int(sys.argv[4])
    except:
        print("Invalid radius and resolution, switching to default : ")
        print("radius = " + str(radius))
        print("resolution = " + str(resolution))

    vertices,faces = GenerateSphericMesh(radius,resolution,save)

    if(info):
        import trimesh

        mesh = trimesh.Trimesh(vertices,faces)
        #mesh.show()

        def getMeshSize(mesh):
            try:
                lines = np.array(mesh.vertices[mesh.edges_unique])
            except:
                mesh = trimesh.Trimesh(mesh.vertices,mesh.faces)
                lines = np.array(mesh.vertices[mesh.edges_unique])
            h = np.linalg.norm(lines[:,1] - lines[:,0],axis=1)

            return(np.min(h),np.max(h),np.average(h),np.std(h))

        def getMeshAreas(mesh):
            try:
                areas = np.array(mesh.area_faces)
            except:
                mesh = trimesh.Trimesh(mesh.vertices,mesh.faces)
                areas = np.array(mesh.area_faces)

            return(np.min(areas),np.max(areas),np.average(areas),np.std(areas))

        print("MESH SIZE - min, max, avg, std : ")
        print(getMeshSize(mesh))

        print("MESH AREAS - min, max, avg, std")
        print(getMeshAreas(mesh))
    