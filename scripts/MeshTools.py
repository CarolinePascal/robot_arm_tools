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
#  @return points, faces Generated mesh points (vertices) and triangular faces (cells, triangles)
def generateSphericMesh(size, resolution, elementType = "P0", saveMesh = False, saveYAML = False):

    try:
        os.makedirs(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/config/meshes/sphere/" + elementType + "/")
    except:
        pass

    if(elementType != "P0" and elementType != "P1"):
        raise NotImplementedError

    #Sphere radius 
    radius = size/2

    #Icosahedron initial number of faces 
    k = 20

    #Tool functions
    T = lambda b,c:  b**2 + c**2 + b*c  #Triangulation number

    alpha = 1
    if(elementType == "P0"):
        alpha = 2 - np.sqrt(1 - (resolution**2)/(3*radius**2))
        
    TTarget = int(np.round((16*np.pi*(alpha*radius)**2)/(k*np.sqrt(3)*(resolution)**2)))

    maxBound = int(np.ceil(np.sqrt(TTarget))) + 1

    solutions = np.empty((maxBound,maxBound))

    for i in range(maxBound):
        for j in range(i,maxBound):
            solutions[i,j] = T(i,j)
            solutions[j,i] = T(i,j)

    #TODO Keep closest solution result-wise
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
        meshPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/config/meshes/sphere/" + elementType + "/" + str(size) + "_" + str(resolution) + ".mesh"
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

        YAMLPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/config/meshes/sphere/" + elementType + "/" + str(size) + "_" + str(resolution) + ".yaml"
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

## Function creating a dual spheric mesh using an icosahedric approximation
#  @param size Size of the sphere as its diameter
#  @param resolution Target resolution of the mesh
#  @param elementType Type of element for triangular faces
#  @param saveMesh Wether to save mesh file or not
#  @param saveYAML Wether to save mesh poses in YAML file or not
#  @return points, faces Generated mesh points (vertices) and triangular faces (cells, triangles)
def generateDualSphericMesh(size, resolution, elementType = "P0", saveMesh = False, saveYAML = False):

    try:
        os.makedirs(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/config/meshes/sphere/" + elementType + "/")
    except:
        pass

    if(elementType != "P0"):
        raise NotImplementedError
    
    points,faces = generateSphericMesh(size,resolution,elementType)
    radius = size/2
    centroids = np.average(points[faces],axis=1) 

    points = centroids
    hull = ConvexHull(points)
    faces = hull.simplices

    if(saveMesh):
        meshPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/config/meshes/sphere/" + elementType + "/dual_" + str(size) + "_" + str(resolution) + ".mesh"
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

        YAMLPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/config/meshes/sphere/" + elementType + "/dual_" + str(size) + "_" + str(resolution) + ".yaml"
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

## Function creating a circular mesh
#  @param size Size of the circle as its diameter
#  @param resolution Target resolution of the mesh
#  @param elementType Type of element for lines
#  @param saveMesh Wether to save mesh file or not
#  @param saveYAML Wether to save mesh poses in YAML file or not
#  @return points, faces Generated mesh points (vertices) and lines (cells)
def generateCircularMesh(size, resolution, elementType = "P1", saveMesh = False, saveYAML = False):

    try:
        os.makedirs(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/config/meshes/circle/" + elementType + "/")
    except:
        pass

    if(elementType != "P1"):
        raise NotImplemented

    pointsNumber = int(np.round(np.pi*size/resolution))
    X = (size/2)*np.cos(np.linspace(0,2*np.pi,pointsNumber,endpoint=False))
    Y = (size/2)*np.sin(np.linspace(0,2*np.pi,pointsNumber,endpoint=False))
    Z = np.zeros(pointsNumber)
    points = np.array([X,Y,Z]).T
    lines = np.vstack((np.arange(pointsNumber),np.roll(np.arange(pointsNumber),-1))).T

    if(saveMesh):
        meshPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/config/meshes/circle/" + elementType + "/" + str(size) + "_" + str(resolution) + ".mesh"
        print("Saving mesh at " + meshPath)
        meshio.write_points_cells(meshPath, list(points), [("line",lines)])

    if(saveYAML):
        raise NotImplemented

    return(points, lines)

## Function plotting a wire representation of a mesh
#  @param vertices Mesh vertices
#  @param faces Mesh faces
#  @param elementType Type of element
#  @param plotVertices Wether to plot the vertices or not
#  @param plotControlPoints Wether to plot the control points or not
#  @param axes Matplotlib axes
#  @param show Wether to display the plot or not
def plotMesh(vertices, faces, elementType = "P0", plotVertices = True, plotControlPoints = True, axes = None, show = True):

    if(axes is None):
        axes = plt.figure().add_subplot(projection='3d')

    if(plotVertices):
        for face in faces:
            facePoints = vertices[face]
            facePoints = np.vstack((facePoints,facePoints[0,:]))
            axes.plot(facePoints[:,0],facePoints[:,1],facePoints[:,2],'k',linewidth='1')

    if(plotControlPoints):
        if (elementType == "P0"):
            centroids = np.average(vertices[faces],axis=1)  #axis 0 : we choose the face, axis 1 : we choose the point, axis 2 : we choose the coordinate
            axes.scatter(centroids[:,0],centroids[:,1],centroids[:,2],marker='o',color='r')
        elif (elementType == "P1"):
            axes.scatter(vertices[:,0],vertices[:,1],vertices[:,2],marker='o',color='r')
        else:
            raise NameError("INVALID ELEMENT TYPE")

    if(show):
        extents = np.array([getattr(axes, 'get_{}lim'.format(dim))() for dim in 'xyz'])
        sz = extents[:,1] - extents[:,0]
        centers = np.mean(extents, axis=1)
        maxsize = max(abs(sz))
        r = maxsize/2
        for ctr, dim in zip(centers, 'xyz'):
            getattr(axes, 'set_{}lim'.format(dim))(ctr - r, ctr + r)
        plt.show()

    return(axes)

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
            
            if(len(lines) == 0):
                h = np.linalg.norm(vertices[faces[:,1]] - vertices[faces[:,0]],axis=1)
            else:
                h = np.linalg.norm(lines[:,1] - lines[:,0],axis=1)
        
        return(np.min(h),np.max(h),np.average(h),np.std(h))

    def getMeshAreas(mesh):
        try:
            areas = np.array(mesh.area_faces)
        except:
            mesh = trimesh.Trimesh(mesh.vertices,mesh.faces)
            areas = np.array(mesh.area_faces)

            if(len(areas) == 0):
                areas = np.zeros(1)

        return(np.min(areas),np.max(areas),np.average(areas),np.std(areas))

    print("MESH RESOLUTION - min, max, avg, std : ")
    print(getMeshResolution(mesh))

    print("MESH AREAS - min, max, avg, std")
    print(getMeshAreas(mesh))

    if elementType == "P0":
        centroids = np.average(vertices[faces],axis=1)  #axis 0 : we choose the face, axis 1 : we choose the point, axis 2 : we choose the coordinate
        distances = []  
        for centroid in centroids:
            distances.extend([np.sort(np.linalg.norm(centroids - centroid, axis=1))[1:4]])  #axis 0 : we choose the point, axis 1 : we choose the coordinate
        distances = np.array(distances)

        print("CONTROL POINTS NUMBER : " + str(len(centroids)))
        print("CONTROL POINTS DISTANCE - min, max, avg, std : ")
        print((np.min(distances),np.max(distances),np.average(distances),np.std(distances)))
        print("CONTROL POINTS ORIGIN DISTANCE - min, max, avg, std : ")
        distancesOrigin = np.linalg.norm(centroids,axis=1)
        print((np.min(distancesOrigin),np.max(distancesOrigin),np.average(distancesOrigin),np.std(distancesOrigin)))
        
    elif elementType == "P1":
        print("CONTROL POINTS NUMBER : " + str(len(vertices)))
        print("CONTROL POINTS DISTANCE - min, max, avg, std : ")
        print(getMeshResolution(mesh))
        print("CONTROL POINTS ORIGIN DISTANCE - min, max, avg, std : ")
        distancesOrigin = np.linalg.norm(vertices,axis=1)
        print((np.min(distancesOrigin),np.max(distancesOrigin),np.average(distancesOrigin),np.std(distancesOrigin)))

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
                #generateDualSphericMesh(size, resolution, elementType, saveMesh, saveYAML)

    elif(meshType == "circle"):
        if((saveMesh and not os.path.isfile(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/config/meshes/" + meshType + "/" + str(size) + "_" + str(resolution) + ".mesh")) or info):
            vertices,faces = generateCircularMesh(size, resolution, elementType, saveMesh, saveYAML)
            if(info):
                getMeshInfo(vertices,faces,elementType)
                plotMesh(vertices,faces,elementType)

    else:
        print("Invalid mesh type !")
        sys.exit(-1)
    
