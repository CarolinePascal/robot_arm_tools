#!/usr/bin/env python3.8

import numpy as np
import os

import subprocess
import anti_lib_progs as anti_lib_progs
from scipy.spatial import ConvexHull
import meshio
import trimesh

import yaml

## Function creating a spheric mesh using an icosahedric approximation
#  @param radius Radius of the sphere
#  @param resolution Target resolution of the mesh
#  @param elementType Type of element for triangular faces
#  @param saveMesh Wether to save mesh file or not
#  @param saveYAML Wether to save mesh poses in YAML file or not
def generateSphericMesh(radius, resolution, elementType = "P0", saveMesh = False, saveYAML = False):

    #Icosahedron initial number of faces 
    k = 20

    #Tool functions
    T = lambda b,c:  b**2 + c**2 + b*c  #Triangulation number

    alpha = 1
    if(elementType == "P0"):
        alpha = 1/np.sqrt(1-(resolution**2)/(4*radius**2))
        
    TTarget = int(np.round((16*np.pi*(alpha*radius)**2)/(k*np.sqrt(3)*resolution**2)))
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

    if(elementType == "P0"):
        hull = ConvexHull(points)
        faces = hull.simplices
        centroids = np.average(points[faces],axis=1)
        delta = radius/np.average(np.linalg.norm(centroids,axis=1))
        points *= delta

    hull = ConvexHull(points)
    faces = hull.simplices

    if(saveMesh):
        meshPath = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/config/meshes/sphere/S_" + str(radius) + "_" + str(resolution) + ".mesh"
        print("Saving mesh at " + meshPath)
        meshio.write_points_cells(meshPath, list(points), [("triangle",list(faces))])

    if(saveYAML):
        meshPoses = np.empty((0,6))
        listPoints = None

        if(elementType == 'P0'):
            listPoints = trimesh.Trimesh(points,faces).triangles_center
            
        elif(elementType == 'P1'):
            listPoints = points

        for point in listPoints:
            inclination, azimuth = np.arccos(point[2]/radius),np.arctan2(point[1],point[0])
            meshPoses = np.vstack((meshPoses,np.hstack((point,[np.pi,inclination,azimuth]))))

        sortedMeshPoses = np.empty((0,6))
        sortedXIndex = np.argsort(meshPoses[:,0])
        sortedX = meshPoses[sortedXIndex,0]
        clusterX = np.linspace(sortedX[0],sortedX[-1],int(np.pi*radius/resolution) + 1)
        for i in range(len(clusterX) - 1):
            pointsIndex = np.where((meshPoses[:,0] > clusterX[i]) & (meshPoses[:,0] <= clusterX[i+1]))[0]
            if(i == 0):
                pointsIndex = np.where((meshPoses[:,0] >= clusterX[i]) & (meshPoses[:,0] <= clusterX[i+1]))[0]
            sortedIndex = np.argsort(np.arctan2(meshPoses[pointsIndex,2],meshPoses[pointsIndex,1]))
            sortedMeshPoses = np.vstack((sortedMeshPoses,meshPoses[pointsIndex[sortedIndex]]))

        YAMLPath = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/config/meshes/sphere/S_" + str(radius) + "_" + str(resolution) + ".yaml"
        print("Saving mesh poses at " + YAMLPath)
        with open(YAMLPath, mode="w+") as file:
            yaml.dump({"elementType":elementType},file)
            yaml.dump({"poses":sortedMeshPoses.tolist()},file)

        #import matplotlib.pyplot as plt
        #ax = plt.axes(projection='3d')
        #scatter = ax.scatter(sortedMeshPoses[:,0],sortedMeshPoses[:,1],sortedMeshPoses[:,2],c=np.#arange(len(sortedMeshPoses)),cmap='jet')
        #ax.set_xlabel("x")
        #ax.set_ylabel("y")
        #ax.set_zlabel("z")
        #plt.colorbar(scatter)
        #plt.show()

    return(points, faces)

## Function displaying mesh information
#  @param vertices Mesh vertices
#  @param faces Mesh faces
def getMeshInfo(vertices,faces):
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

if __name__ == "__main__":
    import sys

    meshType = "sphere"
    radius = 0.1
    resolution = 0.01
    elementType = "P0"
    saveMesh = 0
    saveYAML = 0
    info = 0

    try:
        meshType = sys.argv[1]
        radius = float(sys.argv[2])
        resolution = float(sys.argv[3])
        elementType = sys.argv[4]
        saveMesh = int(sys.argv[5])
        saveYAML = int(sys.argv[6])
        info = int(sys.argv[7])
    except:
        print("Invalid radius and resolution, switching to default : ")
        print("mesh type = " + meshType)
        print("radius = " + str(radius))
        print("resolution = " + str(resolution))
        print("element type = " + elementType)

    if((saveMesh and not os.path.isfile(os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/config/meshes/sphere/S_" + str(radius) + "_" + str(resolution) + ".mesh")) or (saveYAML and not os.path.isfile(os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/config/meshes/sphere/S_" + str(radius) + "_" + str(resolution) + ".yaml")) or info):
        vertices,faces = generateSphericMesh(radius, resolution, elementType, saveMesh, saveYAML)
        if(info):
            getMeshInfo(vertices,faces)
    