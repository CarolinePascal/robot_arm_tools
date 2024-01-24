#!/usr/bin/env python3.8

#Utility packages
import numpy as np
import os
import subprocess
import yaml
from copy import copy, deepcopy

#Mesh packages
import anti_lib_progs as anti_lib_progs
from scipy.spatial import ConvexHull
import meshio
import trimesh

#Plotting packages
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d
import open3d as o3d
import plotly.graph_objects as go

## Function creating a spheric mesh using an icosahedric approximation
#  @param size Size of the sphere as its diameter
#  @param resolution Target resolution of the mesh
#  @param elementType Type of element for triangular faces
#  @param saveMesh Wether to save mesh file or not
#  @param saveYAML Wether to save mesh poses in YAML file or not
#  @param gradientOffset If saveYAML is True, adds additionnal one in two measurements points for gradient computation with given normal offset
#  @return points, faces Generated mesh points (vertices) and triangular faces (cells, triangles)
def generateSphericMesh(size, resolution, elementType = "P0", saveMesh = False, saveYAML = False, gradientOffset = 0.0):

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

        #Build 3D poses (position + orientation) from mesh surfacic 3D points
        for point in listPoints:
            inclination, azimuth = np.arccos(point[2]/radius),np.arctan2(point[1],point[0])
            meshPoses = np.vstack((meshPoses,np.hstack((point,[np.pi,inclination,azimuth]))))

        #Sort 3D poses accoirding to inclination: from top to bottom
        sortedMeshPoses = np.empty((0,6))
        sortedMeshPosesInclination = meshPoses[np.argsort(meshPoses[:,4])]

        #Define a slicing range : we split the sphere vertically according to resolution, and add points by increasing azimuth
        inclinationRange = np.arange(0,np.pi,resolution/(2*radius))

        minBound = inclinationRange[0]
        for maxBound in inclinationRange[1:]:
            localIndex = np.where((sortedMeshPosesInclination[:,4] >= minBound) & (sortedMeshPosesInclination[:,4] < maxBound))[0]   
            localMeshPoses = sortedMeshPosesInclination[localIndex]
            localSortedMeshPosesAzimuth = localMeshPoses[np.argsort(localMeshPoses[:,5])]
            sortedMeshPoses = np.vstack((sortedMeshPoses,localSortedMeshPosesAzimuth))
            minBound = maxBound

        #Because we opted for < maxBound, we have to ensure that no point was left behind for == maxBound
        sortedMeshPoses = np.vstack((sortedMeshPoses,sortedMeshPosesInclination[np.where(sortedMeshPoses[:,4] == inclinationRange[-1])[0]]))

        #Adding normally offseted poses for gradient measurements/computation
        if(gradientOffset != 0.0):
            sortedMeshPosesGradient = sortedMeshPoses
            for (gradientPose, meshPose) in zip(sortedMeshPosesGradient,sortedMeshPoses):
                gradientPose[0] += gradientOffset*np.sin(meshPose[4])*np.cos(meshPose[5])
                gradientPose[1] += gradientOffset*np.sin(meshPose[4])*np.sin(meshPose[5])
                gradientPose[2] += gradientOffset*np.cos(meshPose[4])
            np.insert(sortedMeshPoses,1+np.arange(len(sortedMeshPoses)),sortedMeshPosesGradient,axis=0)

        YAMLPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/config/meshes/sphere/" + elementType + "/" + str(size) + "_" + str(resolution) + ".yaml"
        print("Saving mesh poses at " + YAMLPath)
        with open(YAMLPath, mode="w+") as file:
            yaml.dump({"elementType":elementType},file)
            yaml.dump({"poses":sortedMeshPoses.tolist()},file)
            yaml.dump({"gradientOffset":gradientOffset},file)

    return(points, faces)

## Function creating a dual spheric mesh using an icosahedric approximation
#  @param size Size of the sphere as its diameter
#  @param resolution Target resolution of the mesh
#  @param elementType Type of element for triangular faces
#  @param saveMesh Wether to save mesh file or not
#  @param saveYAML Wether to save mesh poses in YAML file or not
#  @param gradientOffset If saveYAML is True, adds additionnal one in two measurements points for gradient computation with given normal offset
#  @return points, faces Generated mesh points (vertices) and triangular faces (cells, triangles)
def generateDualSphericMesh(size, resolution, elementType = "P0", saveMesh = False, saveYAML = False, gradientOffset = 0.0):

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
        listPoints = points #P1 elements

        #Build 3D poses (position + orientation) from mesh surfacic 3D points
        for point in listPoints:
            inclination, azimuth = np.arccos(point[2]/radius),np.arctan2(point[1],point[0])
            meshPoses = np.vstack((meshPoses,np.hstack((point,[np.pi,inclination,azimuth]))))

        #Sort 3D poses accoirding to inclination: from top to bottom
        sortedMeshPoses = np.empty((0,6))
        sortedMeshPosesInclination = meshPoses[np.argsort(meshPoses[:,4])]

        #Define a slicing range : we split the sphere vertically according to resolution, and add points by increasing azimuth
        inclinationRange = np.arange(0,np.pi,resolution/(2*radius))

        minBound = inclinationRange[0]
        for maxBound in inclinationRange[1:]:
            localIndex = np.where((sortedMeshPosesInclination[:,4] >= minBound) & (sortedMeshPosesInclination[:,4] < maxBound))[0]   
            localMeshPoses = sortedMeshPosesInclination[localIndex]
            localSortedMeshPosesAzimuth = localMeshPoses[np.argsort(localMeshPoses[:,5])]
            sortedMeshPoses = np.vstack((sortedMeshPoses,localSortedMeshPosesAzimuth))
            minBound = maxBound

        #Because we opted for < maxBound, we have to ensure that no point was left behind for == maxBound
        sortedMeshPoses = np.vstack((sortedMeshPoses,sortedMeshPosesInclination[np.where(sortedMeshPoses[:,4] == inclinationRange[-1])[0]]))

        #Adding normally offseted poses for gradient measurements/computation
        if(gradientOffset != 0.0):
            sortedMeshPosesGradient = sortedMeshPoses
            for (gradientPose, meshPose) in zip(sortedMeshPosesGradient,sortedMeshPoses):
                gradientPose[0] += gradientOffset*np.sin(meshPose[4])*np.cos(meshPose[5])
                gradientPose[1] += gradientOffset*np.sin(meshPose[4])*np.sin(meshPose[5])
                gradientPose[2] += gradientOffset*np.cos(meshPose[4])
            np.insert(sortedMeshPoses,1+np.arange(len(sortedMeshPoses)),sortedMeshPosesGradient,axis=0)

        YAMLPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/config/meshes/sphere/" + elementType + "/dual_" + str(size) + "_" + str(resolution) + ".yaml"
        print("Saving mesh poses at " + YAMLPath)
        with open(YAMLPath, mode="w+") as file:
            yaml.dump({"elementType":elementType},file)
            yaml.dump({"poses":sortedMeshPoses.tolist()},file)
            yaml.dump({"gradientOffset":gradientOffset},file)

    return(points, faces)

## Function creating a circular mesh
#  @param size Size of the circle as its diameter
#  @param resolution Target resolution of the mesh
#  @param elementType Type of element for lines
#  @param saveMesh Wether to save mesh file or not
#  @param saveYAML Wether to save mesh poses in YAML file or not
#  @param gradientOffset If saveYAML is True, adds additionnal one in two measurements points for gradient computation with given normal offset
#  @return points, faces Generated mesh points (vertices) and lines (cells)
def generateCircularMesh(size, resolution, elementType = "P1", saveMesh = False, saveYAML = False, gradientOffset = 0.0):

    try:
        os.makedirs(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/config/meshes/circle/" + elementType + "/")
    except:
        pass

    if(elementType != "P1"):
        raise NotImplemented
    
    #Circle radius 
    radius = size/2

    pointsNumber = int(np.round(np.pi*size/resolution))
    X = radius*np.cos(np.linspace(0,2*np.pi,pointsNumber,endpoint=False))
    Y = radius*np.sin(np.linspace(0,2*np.pi,pointsNumber,endpoint=False))
    Z = np.zeros(pointsNumber)
    points = np.array([X,Y,Z]).T
    lines = np.vstack((np.arange(pointsNumber),np.roll(np.arange(pointsNumber),-1))).T

    if(saveMesh):
        meshPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/config/meshes/circle/" + elementType + "/" + str(size) + "_" + str(resolution) + ".mesh"
        print("Saving mesh at " + meshPath)
        meshio.write_points_cells(meshPath, list(points), [("line",lines)])

    if(saveYAML):
        meshPoses = np.empty((0,6))
        listPoints = points #P1 elements

        #Build 3D poses (position + orientation) from mesh surfacic 3D points
        for point in listPoints:
            inclination, azimuth = np.arccos(point[2]/radius),np.arctan2(point[1],point[0])
            meshPoses = np.vstack((meshPoses,np.hstack((point,[np.pi,inclination,azimuth]))))

        #Adding normally offseted poses for gradient measurements/computation
        if(gradientOffset != 0.0):
            meshPosesGradient = meshPoses
            for (gradientPose, meshPose) in zip(meshPosesGradient,meshPoses):
                gradientPose[0] += gradientOffset*np.sin(meshPose[4])*np.cos(meshPose[5])
                gradientPose[1] += gradientOffset*np.sin(meshPose[4])*np.sin(meshPose[5])
                gradientPose[2] += gradientOffset*np.cos(meshPose[4])
            np.insert(meshPoses,1+np.arange(len(meshPoses)),meshPosesGradient,axis=0)

        YAMLPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/config/meshes/circle/" + elementType + "/" + str(size) + "_" + str(resolution) + ".yaml"
        print("Saving mesh poses at " + YAMLPath)
        with open(YAMLPath, mode="w+") as file:
            yaml.dump({"elementType":elementType},file)
            yaml.dump({"poses":meshPoses.tolist()},file)
            yaml.dump({"gradientOffset":gradientOffset},file)

    return(points, lines)

## Function plotting a mesh with P0/P1 elements
#  @param vertices Mesh vertices
#  @param faces Mesh faces
#  @param elementType Type of element
#  @param plotEdges Wether to plot the edges or not
#  @param plotNodes Wether to plot the elements nodes or not
#  @param ax Matplotlib axes
def plotMesh(vertices, faces, elementType = "P0", plotEdges = True, plotNodes = False, ax = None, interactive = False, **kwargs):

    if(ax is None):
        if(interactive):
            ax = go.Figure()
        else:
           _,ax = plt.subplots(1,subplot_kw=dict(projection='3d'))

    if(interactive):
        if(plotEdges):
            kwargsEdges = deepcopy(kwargs)
            kwargsEdges["mode"] = "lines"
            kwargsEdges["line"] = {"color":'rgba(0, 0, 0, 0.1)',"width":5}

            Xe = []
            Ye = []
            Ze = []
            for T in vertices[faces]:
                Xe += [T[k%3][0] for k in range(4)]+[None]
                Ye += [T[k%3][1] for k in range(4)]+[None]
                Ze += [T[k%3][2] for k in range(4)]+[None]

            ax.add_trace(go.Scatter3d(x=Xe,y=Ye,z=Ze,**kwargsEdges))
        if(plotNodes):
            kwargsNodes = deepcopy(kwargs)
            kwargsNodes["mode"] = "markers"
            kwargsNodes["marker"] = {"size":2}
            if (elementType == "P0"):
                centroids = np.average(vertices[faces],axis=1)
                ax.add_trace(go.Scatter3d(x=centroids[:,0],y=centroids[:,1],z=centroids[:,2],**kwargsNodes))
            elif(elementType == "P1"):
                ax.add_trace(go.Scatter3d(x=vertices[:,0],y=vertices[:,1],z=vertices[:,2],**kwargsNodes))
            else:
                print("Invalid element type, nodes will not be displayed")

    else:
        if(plotEdges):
            kwargsEdges = deepcopy(kwargs)
            kwargsEdges["facecolor"] = (0,0,0,0)
            if(not "edgecolor" in kwargsEdges):
                kwargsEdges["edgecolor"] = (0,0,0,0.1)
            plotMesh = art3d.Poly3DCollection(vertices[faces], **kwargsEdges)
            ax.add_collection3d(copy(plotMesh))

        if(plotNodes):
            kwargsNodes = deepcopy(kwargs)
            kwargsNodes["marker"] = "o"
            if (elementType == "P0"):
                centroids = np.average(vertices[faces],axis=1)
                ax.scatter(centroids[:,0],centroids[:,1],centroids[:,2], **kwargsNodes)
            elif (elementType == "P1"):
                ax.scatter(vertices[:,0],vertices[:,1],vertices[:,2], **kwargsNodes)
            else:
                print("Invalid element type, nodes will not be displayed")

        #Set 3D plot limits and aspect
        xlim = ax.get_xlim()
        deltaX = xlim[1] - xlim[0]
        meanX = np.mean(xlim)
        ylim = ax.get_ylim()
        deltaY = ylim[1] - ylim[0]
        meanY = np.mean(ylim)
        zlim = ax.get_zlim()
        deltaZ = zlim[1] - zlim[0]
        meanZ = np.mean(zlim)

        delta = np.max([deltaX,deltaY,deltaZ])

        ax.set_xlim(meanX - 0.5*delta, meanX + 0.5*delta)
        ax.set_ylim(meanY - 0.5*delta, meanY + 0.5*delta)
        ax.set_zlim(meanZ - 0.5*delta, meanZ + 0.5*delta)

        ax.set_box_aspect((1,1,1))

    return(ax)

def plotMeshFromPath(meshPath, elementType = "P0", plotEdges = True, plotNodes = False, ax = None, interactive = False, **kwargs):
	mesh = meshio.read(meshPath)
	vertices, faces = mesh.points, mesh.get_cells_type("triangle")
	return(plotMesh(vertices, faces, elementType, plotEdges, plotNodes, ax, interactive, **kwargs))

## Function plotting a colored point cloud
#  @param points Point cloud points
#  @param colors Point cloud points colors
#  @param ax Matplotlib axes
def plotPointCloud(points, colors = None, ax = None, interactive = False, **kwargs):
    if(ax is None):
        if(interactive):
            ax = go.Figure()
        else:
            _,ax = plt.subplots(1,subplot_kw=dict(projection='3d'))

    if(interactive):
        kwargsPointCloud = deepcopy(kwargs)
        kwargsPointCloud["mode"] = "markers"
        kwargsPointCloud["marker"] = {"color":colors,"size":2}

        ax.add_trace(go.Scatter3d(x=points[:,0],y=points[:,1],z=points[:,2],**kwargsPointCloud))
    else:
        kwargsPointCloud = deepcopy(kwargs)
        kwargsPointCloud["c"] = colors

        if(not "s" in kwargsPointCloud):
            kwargsPointCloud["s"] = 2

        ax.scatter(*points.T, **kwargsPointCloud)

        #Set 3D plot limits and aspect
        xlim = ax.get_xlim()
        deltaX = xlim[1] - xlim[0]
        meanX = np.mean(xlim)
        ylim = ax.get_ylim()
        deltaY = ylim[1] - ylim[0]
        meanY = np.mean(ylim)
        zlim = ax.get_zlim()
        deltaZ = zlim[1] - zlim[0]
        meanZ = np.mean(zlim)

        delta = np.max([deltaX,deltaY,deltaZ])

        ax.set_xlim(meanX - 0.5*delta, meanX + 0.5*delta)
        ax.set_ylim(meanY - 0.5*delta, meanY + 0.5*delta)
        ax.set_zlim(meanZ - 0.5*delta, meanZ + 0.5*delta)

        ax.set_box_aspect((1,1,1))
        
    return(ax)

def plotPointCloudFromPath(pointCloudPath, ax = None, interactive = False, **kwargs):
    pointCloud = o3d.io.read_point_cloud(pointCloudPath) 
    pointCloud = pointCloud.voxel_down_sample(voxel_size=0.005)
    points = np.array(pointCloud.points)
    colors = np.array(pointCloud.colors)
    return(plotPointCloud(points, colors, ax, interactive, **kwargs))

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
    gradientOffset = 0.0
    info = 0

    try:
        meshType = sys.argv[1]
        size = float(sys.argv[2])
        resolution = float(sys.argv[3])
        elementType = sys.argv[4]
        saveMesh = int(sys.argv[5])
        saveYAML = int(sys.argv[6])
        gradientOffset = float(sys.argv[7])
        info = int(sys.argv[8])
    except:
        print("Invalid size and resolution, switching to default : ")
        print("mesh type = " + meshType)
        print("size = " + str(size))
        print("resolution = " + str(resolution))
        print("element type = " + elementType)

    if(meshType == "sphere"):
        if((saveMesh and not os.path.isfile(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/config/meshes/" + meshType + "/" + elementType + "/" + str(size) + "_" + str(resolution) + ".mesh")) or (saveYAML and not os.path.isfile(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/config/meshes/" + meshType + "/" + elementType + "/" + str(size) + "_" + str(resolution) + ".yaml")) or info):
            vertices,faces = generateSphericMesh(size, resolution, elementType, saveMesh, saveYAML, gradientOffset)
            if(info):
                getMeshInfo(vertices,faces,elementType)
                plotMesh(vertices,faces,elementType)
                #generateDualSphericMesh(size, resolution, elementType, saveMesh, saveYAML, gradientOffset)

    elif(meshType == "circle"):
        if((saveMesh and not os.path.isfile(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/config/meshes/" + meshType + "/" + elementType + "/" + str(size) + "_" + str(resolution) + ".mesh")) or info):
            vertices,faces = generateCircularMesh(size, resolution, elementType, saveMesh, saveYAML, gradientOffset)
            if(info):
                getMeshInfo(vertices,faces,elementType)
                plotMesh(vertices,faces,elementType)

    else:
        print("Invalid mesh type !")
        sys.exit(-1)
    
