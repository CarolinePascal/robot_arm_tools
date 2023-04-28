#!/usr/bin/env python3.8

import numpy as np
import matplotlib.pyplot as plt
import os

import subprocess
import anti_lib_progs as anti_lib_progs
from scipy.spatial import ConvexHull
import meshio
import trimesh

import yaml

## Function creating a spheric mesh using an icosahedric approximation
#  @param size Size of the sphere as its diameter
#  @param resolution Target resolution of the mesh
#  @param elementType Type of element for triangular faces
#  @param saveMesh Wether to save mesh file or not
#  @param saveYAML Wether to save mesh poses in YAML file or not
def generateSphericMesh(size, resolution, elementType = "P0", saveMesh = False, saveYAML = False):

    #Sphere radius 
    radius = size/2

    #Icosahedron initial number of faces 
    k = 20

    #Tool functions
    T = lambda b,c:  b**2 + c**2 + b*c  #Triangulation number

    #Not sure about this..
    beta = 1
    #if (elementType == "P0"):
        #beta = np.sqrt(3)

    alpha = 1
    if(elementType == "P0"):
        alpha = 1/np.sqrt(1-((beta*resolution)**2)/(4*radius**2))
        
    TTarget = int(np.round((16*np.pi*(alpha*radius)**2)/(k*np.sqrt(3)*(beta*resolution)**2)))
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
        centroids = np.average(points[faces],axis=1)    #axis 0 : we choose the face, axis 1 : we choose the point, axis 2 : we choose the coordinate
        delta = radius/np.average(np.linalg.norm(centroids,axis=1)) #axis 0 : we choose the face, axis 1 : we choose the coordinate
        points *= delta

    hull = ConvexHull(points)
    faces = hull.simplices

    if(saveMesh):
        meshPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/config/meshes/sphere/" + str(size) + "_" + str(resolution) + ".mesh"
        print("Saving mesh at " + meshPath)
        meshio.write_points_cells(meshPath, list(points), [("triangle",list(faces))])

    if(saveYAML):
        meshPoses = np.empty((0,6))
        listPoints = None

        if(elementType == 'P0'):
            listPoints = np.average(points[faces],axis=1)   #axis 0 : we choose the face, axis 1 : we choose the point, axis 2 : we choose the coordinate
            
        elif(elementType == 'P1'):
            listPoints = points

        for point in listPoints:
            inclination, azimuth = np.arccos(point[2]/radius),np.arctan2(point[1],point[0])
            meshPoses = np.vstack((meshPoses,np.hstack((point,[np.pi,inclination,azimuth]))))

        sortedMeshPoses = np.empty((0,6))

        sortedMeshPosesInclination = meshPoses[np.argsort(meshPoses[:,4])]

        inclinationRange = np.arange(0,np.pi,resolution/(2*radius))

        minBound = inclinationRange[0]
        for maxBound in inclinationRange[1:]:
            localIndex = np.where((sortedMeshPosesInclination[:,4] >= minBound) & (sortedMeshPosesInclination[:,4] < maxBound))[0]   
            localMeshPoses = sortedMeshPosesInclination[localIndex]
            localSortedMeshPosesAzimuth = localMeshPoses[np.argsort(localMeshPoses[:,5])]
            sortedMeshPoses = np.vstack((sortedMeshPoses,localSortedMeshPosesAzimuth))
            minBound = maxBound

        sortedMeshPoses = np.vstack((sortedMeshPoses,sortedMeshPosesInclination[np.where(sortedMeshPoses[:,4] == inclinationRange[-1])[0]]))

        YAMLPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/config/meshes/sphere/" + str(size) + "_" + str(resolution) + ".yaml"
        print("Saving mesh poses at " + YAMLPath)
        with open(YAMLPath, mode="w+") as file:
            yaml.dump({"elementType":elementType},file)
            yaml.dump({"poses":sortedMeshPoses.tolist()},file)

        """
        import matplotlib.pyplot as plt
        ax = plt.axes(projection='3d')
        scatter = ax.scatter(sortedMeshPoses[:200,0],sortedMeshPoses[:200,1],sortedMeshPoses[:200,2],c=np.arange(200),cmap='jet')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.colorbar(scatter)

        extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
        sz = extents[:,1] - extents[:,0]
        centers = np.mean(extents, axis=1)
        maxsize = max(abs(sz))
        r = maxsize/2
        for ctr, dim in zip(centers, 'xyz'):
            getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

        plt.show()
        """

    return(points, faces)

## Function plotting a wire representation of a mesh
#  @param vertices Mesh vertices
#  @param faces Mesh faces
#  @param elementType Type of element
def plotMesh(vertices,faces,elementType):
    ax = plt.figure().add_subplot(projection='3d')

    for face in faces:
        facePoints = vertices[face]
        facePoints = np.vstack((facePoints,facePoints[0,:]))
        ax.plot(facePoints[:,0],facePoints[:,1],facePoints[:,2],'k',linewidth='1')

    if (elementType == "P0"):
        centroids = np.average(vertices[faces],axis=1)  #axis 0 : we choose the face, axis 1 : we choose the point, axis 2 : we choose the coordinate
        ax.scatter(centroids[:,0],centroids[:,1],centroids[:,2],marker='o',color='r')
    elif (elementType == "P1"):
        ax.scatter(vertices[:,0],vertices[:,1],vertices[:,2],marker='o',color='r')

    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)
    plt.show()

## Function displaying mesh information
#  @param vertices Mesh vertices
#  @param faces Mesh faces
#  @param elementType Type of element
def getMeshInfo(vertices,faces,elementType="P0"):

    mesh = trimesh.Trimesh(vertices,faces)

    def getMeshResolution(mesh):
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

    print("MESH RESOLUTION - min, max, avg, std : ")
    print(getMeshResolution(mesh))

    print("MESH AREAS - min, max, avg, std")
    print(getMeshAreas(mesh))

    if elementType == "P0":
        centroids = np.average(vertices[faces],axis=1)  #axis 0 : we choose the face, axis 1 : we choose the point, axis 2 : we choose the coordinate
        distances = []  
        for i,centroid in enumerate(centroids):
            closestNeighbours = [face for face in faces if ((faces[i][0] in face and faces[i][1] in face) or (faces[i][1] in face and faces[i][2] in face) or (faces[i][2] in face and faces[i][0] in face)) and (face != faces[i]).any()]   

            distances.extend([np.linalg.norm(centroid - np.average(vertices[face],axis=0)) for face in closestNeighbours])  #axis 0 : we choose the point, axis 1 : we choose the coordinate

        distances = np.array(distances)

        print("CONTROL POINTS DISTANCE - min, max, avg, std : ")
        print((np.min(distances),np.max(distances),np.average(distances),np.std(distances)))
        
    elif elementType == "P1":
        print("CONTROL POINTS DISTANCE - min, max, avg, std : ")
        print(getMeshResolution(mesh))

if __name__ == "__main__":
    import sys

    meshType = "sphere"
    size = 0.1
    resolution = 0.01
    elementType = "P0"
    saveMesh = 0
    saveYAML = 0
    info = 0

    try:
        meshType = sys.argv[1]
        size = float(sys.argv[2])
        resolution = float(sys.argv[3])
        elementType = sys.argv[4]
        saveMesh = int(sys.argv[5])
        saveYAML = int(sys.argv[6])
        info = int(sys.argv[7])
    except:
        print("Invalid size and resolution, switching to default : ")
        print("mesh type = " + meshType)
        print("size = " + str(size))
        print("resolution = " + str(resolution))
        print("element type = " + elementType)

    if(meshType == "sphere"):
        if((saveMesh and not os.path.isfile(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/config/meshes/" + meshType + "/" + str(size) + "_" + str(resolution) + ".mesh")) or (saveYAML and not os.path.isfile(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/config/meshes/" + meshType + "/" + str(size) + "_" + str(resolution) + ".yaml")) or info):
            vertices,faces = generateSphericMesh(size, resolution, elementType, saveMesh, saveYAML)
            if(info):
                getMeshInfo(vertices,faces,elementType)
                plotMesh(vertices,faces,elementType)

    else:
        print("Invalid mesh type !")
        sys.exit(-1)
    