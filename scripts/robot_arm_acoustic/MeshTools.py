#!/usr/bin/env python3.8

#Utility packages
import numpy as np
import os
import subprocess
import yaml
from copy import deepcopy
import cloup

#Mesh packages
import anti_lib_progs as anti_lib_progs
from scipy.spatial import ConvexHull
import meshio
import trimesh
import dualmesh as dm

#Point cloud packages
import point_cloud_utils as pcu
import open3d as o3d

#Plotting packages
import matplotlib.pyplot as plt
import plotly.graph_objects as go

LEGACY = False
PACKAGE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

## Function computing the best values of the b and c parameters for the icosaedric approximation of a sphere
#  @param size Size of the sphere as its diameter
#  @param resolution Target resolution of the mesh
#  @param elementType Type of element for triangular faces
#  @return b, c Computed b and c parameters
def getTargetParameters(size,resolution,elementType = "P0"):

    radius = size/2

    #Icosahedron initial number of faces 
    k = 20

    #Tool functions
    T = lambda b,c:  b**2 + c**2 + b*c  #Triangulation number

    alpha = 1
    if(elementType == "P0"):
        if(LEGACY):
            alpha = 1/np.sqrt(1-((resolution)**2)/(4*radius**2))
        else:
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
    return(solution[0], solution[1])

## Function creating a spheric mesh using an icosahedric approximation with given b and c parameters
#  @param size Size of the sphere as its diameter
#  @param resolution Target resolution of the mesh
#  @param b Icosahedric approximation b parameter
#  @param c Icosahedric approximation c parameter
#  @param elementType Type of element for triangular faces
#  @return points, faces Generated mesh points (vertices) and triangular faces (cells, triangles)
def generateSphericMeshFromParameters(size, b, c, elementType = "P0"):

    radius = size/2

    args = " -l -o /tmp/points.txt -p i -c " + str(b) + "," + str(c)
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

    vertices = hull.points[hull.vertices]
    faces = hull.simplices

    return(vertices, faces)

## Function saving a spheric mesh poses in a YAML file
#  @param YAMLPath Path to the YAML file
#  @param vertices Mesh vertices
#  @param faces Mesh faces
#  @param size Size of the sphere as its diameter
#  @param resolution Resolution of the mesh
#  @param elementType Type of element for triangular faces
#  @param gradientOffset Adds additionnal one in two measurements points for gradient computation with given normal offset
def saveMeshYAML(YAMLPath, vertices, faces, size, resolution, elementType = "P0", gradientOffset = 0.0):

    #Sphere radius
    radius = size/2

    meshPoses = np.empty((0,6))
    listPoints = None

    if(elementType == 'P0'):
        listPoints = np.average(vertices[faces],axis=1)   #axis 0 : we choose the face, axis 1 : we choose the point, axis 2 : we choose the coordinate
        
    elif(elementType == 'P1'):
        listPoints = vertices

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

    print("Saving mesh poses at " + YAMLPath)
    with open(YAMLPath, mode="w+") as file:
        yaml.dump({"elementType":elementType},file)
        yaml.dump({"poses":sortedMeshPoses.tolist()},file)
        yaml.dump({"gradientOffset":gradientOffset},file)

## Function creating a spheric mesh using an icosahedric approximation
#  @param size Size of the sphere as its diameter
#  @param resolution Target resolution of the mesh
#  @param sigma Mesh vertices position standard deviation
#  @param elementType Type of element for triangular faces
#  @param saveMesh Wether to save mesh file or not
#  @param saveYAML Wether to save mesh poses in YAML file or not
#  @param gradientOffset If saveYAML is True, adds additionnal one in two measurements points for gradient computation with given normal offset
#  @param saveFolder Folder where to save the mesh
#  @return points, faces Generated mesh points (vertices) and triangular faces (cells, triangles)
def generateSphericMesh(size, resolution, sigma = 0.0, elementType = "P0", saveMesh = False, saveYAML = False, gradientOffset = 0.0, saveFolder = None):

    folderName = PACKAGE_PATH + "/config/meshes/sphere" + "_legacy"*LEGACY + "_uncertainty"*(sigma != 0.0) + "/" + elementType
    if(not saveFolder is None):
        folderName = saveFolder + "/sphere" + "_legacy"*LEGACY + "_uncertainty"*(sigma != 0.0) + "/" + elementType
    os.makedirs(folderName, exist_ok=True)

    if(elementType != "P0" and elementType != "P1"):
        raise NotImplementedError

    bTarget, cTarget = getTargetParameters(size,resolution,elementType)
    points, faces = generateSphericMeshFromParameters(size, bTarget, cTarget, elementType)

    if(sigma != 0.0):
        delta = np.random.normal(0,sigma,points.shape)
        points += delta

    if(saveMesh):
        meshPath = folderName + "/" + str(size) + "_" + str(resolution) + ".mesh"
        print("Saving mesh at " + meshPath)
        meshio.write_points_cells(meshPath, list(points), [("triangle",list(faces))])

    if(saveYAML):
        YAMLPath = folderName + "/" + str(size) + "_" + str(resolution) + ".yaml"
        saveMeshYAML(YAMLPath, points, faces, size, resolution, elementType, gradientOffset)

    return(points, faces)

## Function creating a dual spheric mesh using an icosahedric approximation
#  @param size Size of the sphere as its diameter
#  @param resolution Target resolution of the mesh
#  @param sigma Mesh vertices position standard deviation
#  @param elementType Type of element for triangular faces
#  @param saveMesh Wether to save mesh file or not
#  @param saveYAML Wether to save mesh poses in YAML file or not
#  @param gradientOffset If saveYAML is True, adds additionnal one in two measurements points for gradient computation with given normal offset
#  @param saveFolder Folder where to save the mesh
#  @return points, faces Generated mesh points (vertices) and triangular faces (cells, triangles)
def generateDualSphericMesh(size, resolution, sigma = 0.0, elementType = "P0", saveMesh = False, saveYAML = False, gradientOffset = 0.0, saveFolder = None):

    folderName = PACKAGE_PATH + "/config/meshes/sphere" + "_legacy"*LEGACY + "_uncertainty"*(sigma != 0.0) + "/" + elementType
    if(not saveFolder is None):
        folderName = saveFolder + "/sphere" + "_legacy"*LEGACY + "_uncertainty"*(sigma != 0.0) + "/" + elementType
    os.makedirs(folderName, exist_ok=True)

    if(elementType != "P0"):
        raise NotImplementedError
    
    points,faces = generateSphericMesh(size,resolution,sigma,elementType)
    dualMesh = dm.get_dual(meshio.Mesh(points, [("triangle",faces)]), order=True)
    points,faces = dualMesh.points, dualMesh.get_cells_type("polygon")

    if(saveMesh):
        meshPath = folderName + "/" + "dual_" + str(size) + "_" + str(resolution) + ".mesh"
        print("Saving mesh at " + meshPath)
        meshio.write_points_cells(meshPath, list(points), [("polygon",list(faces))])

    if(saveYAML):
        YAMLPath = folderName + "/" + str(size) + "_" + str(resolution) + ".yaml"
        saveMeshYAML(YAMLPath, points, faces, size, resolution, elementType, gradientOffset)

    return(points, faces)

## Function creating similar spheric meshes (meshes with similar elements nodes but larger resolutions)
#  @param size Size of the sphere as its diameter
#  @param resolution Target resolution of the mesh
#  @param sigma Mesh vertices position standard deviation
#  @param elementType Type of element for triangular faces
#  @param saveMesh Wether to save mesh file or not
#  @param saveYAML Wether to save mesh poses in YAML file or not
#  @param gradientOffset If saveYAML is True, adds additionnal one in two measurements points for gradient computation with given normal offset
#  @param saveFolder Folder where to save the mesh
#  @return list((points, faces)) Generated mesh points (vertices) and triangular faces (cells, triangles)
def generateSimilarSphericMeshes(size, resolution, sigma, elementType = "P0", similarNumber = 3, saveMesh = False, saveYAML = False, gradientOffset = 0.0, saveFolder = None):

    vertices,faces = generateSphericMesh(size,resolution,0.0,elementType)
    similarMeshes = []

    folderName = PACKAGE_PATH + "/config/meshes/sphere" + "_legacy"*LEGACY + "_uncertainty"*(sigma != 0.0) + "/" + elementType + "/similar_" + str(size) + "_" + str(resolution)
    if(not saveFolder is None):
        folderName = saveFolder + "/sphere" + "_legacy"*LEGACY + "_uncertainty"*(sigma != 0.0) + "/" + elementType + "/similar_" + str(size) + "_" + str(resolution)
    os.makedirs(folderName, exist_ok=True)
    
    #This is actually a problem that does not always have a solution : some meshes will have similar lower resolution meshes, but others wont have any !
    if(elementType == "P0"):

        #Compute current mesh data               
        bmax,cmax = getTargetParameters(size,resolution,elementType)
        centroidsInit = np.average(vertices[faces],axis=1)

        ### Find the (b,c) parameters combinations leading to the best centroid fit
        centroidsDistances = np.ma.empty((bmax+1,cmax+1))

        for b in range(bmax+1):
            for c in range(cmax+1):

                #Remove current mesh parameters, and redundant parameters
                if((c == cmax and b == bmax) or (c == 0)):
                    centroidsDistances[b,c] = np.ma.masked
                    continue

                #Compute distance from current mesh centroids to the studied mesh centroids
                verticesTmp,facesTmp = generateSphericMeshFromParameters(size,b,c,elementType)
                centroidsTmp = np.average(verticesTmp[facesTmp],axis=1)

                distance = 0
                for centroid in centroidsTmp:
                    tmp = np.min(np.linalg.norm(centroidsInit - centroid, axis=1))
                    if(tmp > distance):
                        distance = tmp

                centroidsDistances[b,c] = distance

        #Keep only the best mesh which are the closest centroid-wise to the current mesh
        centroidsDistances = np.ma.array(centroidsDistances, mask=np.isnan(centroidsDistances))
        bestParameters = np.array(np.ma.where(centroidsDistances <= 0.5*np.std(centroidsDistances))).T
        
        #Tweak the best meshes for better centroid fit
        for parameters in bestParameters:
            verticesTmp,facesTmp = generateSphericMeshFromParameters(size,parameters[0],parameters[1],elementType)
            centroidsTmp = np.average(verticesTmp[facesTmp],axis=1)

            #Compute M s.t. centroidsTmp = M*verticesTmp
            M = np.zeros((3*len(facesTmp),3*len(verticesTmp)))
            for i,face in enumerate(facesTmp):
                for j,index in enumerate(face):
                    M[3*i:3*i+3,3*index:3*index+3] = np.eye(3)/3

            #Get the matching (closest) centroids in the initial mesh
            centroidsInitMatch = np.zeros(centroidsTmp.shape)
            for i,centroid in enumerate(centroidsTmp):
                index = np.argmin(np.linalg.norm(centroidsInit - centroid, axis=1))
                centroidsInitMatch[i] = centroidsInit[index]

            #Find verticesTmp s.t. |centroidsTmp - centroidsInitMatch| = |M*verticesTmp - centroidsInitMatch| is minimized
            verticesTmp = np.linalg.lstsq(M,centroidsInitMatch.flatten(),rcond=None)[0].reshape(-1,3)

            #Get new mesh resolution
            resolutionTmp = float(trimesh.Trimesh(verticesTmp,facesTmp).edges_unique_length.max())
            resolutionTmp = np.round(resolutionTmp,3)

            if(sigma != 0.0):
                delta = np.random.normal(0,sigma,verticesTmp.shape)
                verticesTmp += delta

            if(saveMesh):
                meshPath = folderName + "/" + str(size) + "_" + str(resolutionTmp) + ".mesh"
                print("Saving mesh at " + meshPath)
                meshio.write_points_cells(meshPath, list(verticesTmp), [("triangle",list(facesTmp))])

            if(saveYAML):
                YAMLPath = folderName + "/" + str(size) + "_" + str(resolutionTmp) + ".yaml"
                saveMeshYAML(YAMLPath, verticesTmp, facesTmp, size, resolutionTmp, elementType, gradientOffset)

            similarMeshes.append((verticesTmp,facesTmp))

    #This problem however does always have a solution :)
    if(elementType == "P1"):
        
        similarResolutions = [resolution*(1+i) for i in range(similarNumber)]

        for similarResolution in similarResolutions:
            idx = pcu.downsample_point_cloud_poisson_disk(vertices, similarResolution)
            sampledPoints = vertices[idx]

            hull = ConvexHull(sampledPoints)
            verticesTmp, facesTmp = hull.points[hull.vertices], hull.simplices

            #Get new mesh resolution
            resolutionTmp = float(trimesh.Trimesh(verticesTmp,facesTmp).edges_unique_length.max())
            resolutionTmp = np.round(resolutionTmp,3)

            if(sigma != 0.0):
                delta = np.random.normal(0,sigma,verticesTmp.shape)
                verticesTmp += delta

            if(saveMesh):
                meshPath = folderName + "/" + str(size) + "_" + str(resolutionTmp) + ".mesh"
                print("Saving mesh at " + meshPath)
                meshio.write_points_cells(meshPath, list(verticesTmp), [("triangle",list(facesTmp))])

            if(saveYAML):
                YAMLPath = folderName + "/" + str(size) + "_" + str(resolutionTmp) + ".yaml"
                saveMeshYAML(YAMLPath, verticesTmp, facesTmp, size, resolutionTmp, elementType, gradientOffset)

            similarMeshes.append((verticesTmp,facesTmp))

    return(similarMeshes)

## Function creating a circular mesh
#  @param size Size of the circle as its diameter
#  @param resolution Target resolution of the mesh
#  @param sigma Mesh vertices position standard deviation
#  @param elementType Type of element for lines
#  @param saveMesh Wether to save mesh file or not
#  @param saveYAML Wether to save mesh poses in YAML file or not
#  @param gradientOffset If saveYAML is True, adds additionnal one in two measurements points for gradient computation with given normal offset
#  @param saveFolder Folder where to save the mesh
#  @return points, faces Generated mesh points (vertices) and lines (cells)
def generateCircularMesh(size, resolution, sigma = 0.0, elementType = "P1", saveMesh = False, saveYAML = False, gradientOffset = 0.0, saveFolder = None):

    folderName = PACKAGE_PATH + "/config/meshes/circle" + "_uncertainty"*(sigma != 0.0) + "/" + elementType
    if(not saveFolder is None):
        folderName = saveFolder + "/circle" + "_uncertainty"*(sigma != 0.0) + "/" + elementType
    os.makedirs(folderName, exist_ok=True)

    if(elementType != "P1"):
        raise NotImplementedError
    
    #Circle radius 
    radius = size/2

    pointsNumber = int(np.round(np.pi*size/resolution))
    X = radius*np.cos(np.linspace(0,2*np.pi,pointsNumber,endpoint=False))
    Y = radius*np.sin(np.linspace(0,2*np.pi,pointsNumber,endpoint=False))
    Z = np.zeros(pointsNumber)
    points = np.array([X,Y,Z]).T
    lines = np.vstack((np.arange(pointsNumber),np.roll(np.arange(pointsNumber),-1))).T

    if(sigma != 0.0):
        delta = np.random.normal(0,sigma,points.shape)
        points += delta

    if(saveMesh):
        meshPath = folderName + "/" + str(size) + "_" + str(resolution) + ".mesh"
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

        YAMLPath = folderName + "/" + str(size) + "_" + str(resolution) + ".yaml"

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
            kwargsEdges["line"] = {}
            kwargsEdges["line"]["color"] = "rgba" + str(kwargsEdges.pop("edgecolor",(0,0,0,0.2)))
            kwargsEdges["line"]["width"] = kwargsEdges.pop("linewidth",5)

            if(not kwargsEdges.pop("facecolor",None) is None):
                print("Facecolor will not be displayed in interactive mode")

            Xe = []
            Ye = []
            Ze = []
            for T in vertices[faces]:
                Xe += [T[k%3][0] for k in range(4)]+[None]
                Ye += [T[k%3][1] for k in range(4)]+[None]
                Ze += [T[k%3][2] for k in range(4)]+[None]

            ax.add_trace(go.Scatter3d(x=Xe,y=Ye,z=Ze,hoverinfo='skip',**kwargsEdges))

        if(plotNodes):
            kwargsNodes = deepcopy(kwargs)
            kwargsNodes["mode"] = "markers"
            kwargsNodes["marker_symbol"] = "circle"
            kwargsNodes["marker"] = {}
            kwargsNodes["marker"]["size"] = kwargsNodes.pop("s",4)
            kwargsNodes["marker"]["color"] = "rgba" + str(kwargsNodes.pop("c",(1.0,0.0,0.0)))

            if (elementType == "P0"):
                centroids = np.average(vertices[faces],axis=1)
                ax.add_trace(go.Scatter3d(x=centroids[:,0],y=centroids[:,1],z=centroids[:,2],hoverinfo='skip',**kwargsNodes))
            elif(elementType == "P1"):
                ax.add_trace(go.Scatter3d(x=vertices[:,0],y=vertices[:,1],z=vertices[:,2],hoverinfo='skip',**kwargsNodes))
            else:
                print("Invalid element type, nodes will not be displayed")

    else:
        if(plotEdges):
            kwargsEdges = deepcopy(kwargs)
            if(not "facecolor" in kwargsEdges):
                kwargsEdges["facecolor"] = (0,0,0.0,0)
            if(not "edgecolor" in kwargsEdges):
                kwargsEdges["edgecolor"] = (0,0,0,0.1)
            if(not "linewidth" in kwargsEdges):
                kwargsEdges["linewidth"] = 2
            ax.plot_trisurf(vertices[:,0],vertices[:,1],vertices[:,2],triangles=faces, **kwargsEdges)
            
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
        xlim = ax.get_xlim3d()
        deltaX = xlim[1] - xlim[0]
        meanX = np.mean(xlim)
        ylim = ax.get_ylim3d()
        deltaY = ylim[1] - ylim[0]
        meanY = np.mean(ylim)
        zlim = ax.get_zlim3d()
        deltaZ = zlim[1] - zlim[0]
        meanZ = np.mean(zlim)

        delta = np.max([deltaX,deltaY,deltaZ])

        ax.set_xlim3d(meanX - 0.5*delta, meanX + 0.5*delta)
        ax.set_ylim3d(meanY - 0.5*delta, meanY + 0.5*delta)
        ax.set_zlim3d(meanZ - 0.5*delta, meanZ + 0.5*delta)

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
        kwargsPointCloud["marker_symbol"] = "circle"
        kwargsPointCloud["marker"] = {}
        kwargsPointCloud["marker"]["color"] = colors
        kwargsPointCloud["marker"]["size"] = kwargsPointCloud.pop("s",2)

        ax.add_trace(go.Scatter3d(x=points[:,0],y=points[:,1],z=points[:,2],hoverinfo='skip',**kwargsPointCloud))
    else:
        kwargsPointCloud = deepcopy(kwargs)
        kwargsPointCloud["c"] = colors

        if(not "s" in kwargsPointCloud):
            kwargsPointCloud["s"] = 5

        ax.scatter(*points.T, **kwargsPointCloud)

        #Set 3D plot limits and aspect
        xlim = ax.get_xlim3d()
        deltaX = xlim[1] - xlim[0]
        meanX = np.mean(xlim)
        ylim = ax.get_ylim3d()
        deltaY = ylim[1] - ylim[0]
        meanY = np.mean(ylim)
        zlim = ax.get_zlim3d()
        deltaZ = zlim[1] - zlim[0]
        meanZ = np.mean(zlim)

        delta = np.max([deltaX,deltaY,deltaZ])

        ax.set_xlim3d(meanX - 0.5*delta, meanX + 0.5*delta)
        ax.set_ylim3d(meanY - 0.5*delta, meanY + 0.5*delta)
        ax.set_zlim3d(meanZ - 0.5*delta, meanZ + 0.5*delta)

        ax.set_box_aspect((1,1,1))
        
    return(ax)

def plotPointCloudFromPath(pointCloudPath, ax = None, interactive = False, **kwargs):
    pointCloud = o3d.io.read_point_cloud(pointCloudPath) 
    pointCloud = pointCloud.voxel_down_sample(voxel_size=0.005)
    points = np.array(pointCloud.points)
    colors = np.array(pointCloud.colors)
    return(plotPointCloud(points, colors, ax, interactive, **kwargs))

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

## Function displaying mesh information
#  @param vertices Mesh vertices
#  @param faces Mesh faces
#  @param elementType Type of element
def getMeshInfo(vertices,faces,elementType="P0"):

    mesh = trimesh.Trimesh(vertices,faces)

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

        print("ELEMENTS NODES NUMBER : " + str(len(centroids)))
        print("ELEMENTS NODES DISTANCE - min, max, avg, std : ")
        print((np.min(distances),np.max(distances),np.average(distances),np.std(distances)))
        print("ELEMENTS NODES ORIGIN DISTANCE - min, max, avg, std : ")
        distancesOrigin = np.linalg.norm(centroids,axis=1)
        print((np.min(distancesOrigin),np.max(distancesOrigin),np.average(distancesOrigin),np.std(distancesOrigin)))
        
    elif elementType == "P1":
        print("ELEMENTS NODES NUMBER : " + str(len(vertices)))
        print("ELEMENTS NODES DISTANCE - min, max, avg, std : ")
        print(getMeshResolution(mesh))
        print("ELEMENTS NODES ORIGIN DISTANCE - min, max, avg, std : ")
        distancesOrigin = np.linalg.norm(vertices,axis=1)
        print((np.min(distancesOrigin),np.max(distancesOrigin),np.average(distancesOrigin),np.std(distancesOrigin)))


@cloup.command()
@cloup.option("--mesh_type", type=str, default="sphere", help="Type of mesh to generate")
@cloup.option("--size", type=float, default=0.1, help="Size (dimension) of the mesh")
@cloup.option("--resolution", type=float, default=0.01, help="Resolution of the mesh")
@cloup.option("--sigma", type=float, default=0.0, help="Position error standard deviation of the mesh vertices")
@cloup.option("--element_type", type=str, default="P0", help="Type of element for triangular faces")
@cloup.option("--save_mesh", is_flag=True, help="Saves the mesh in a .mesh file")
@cloup.option("--save_yaml", is_flag=True, help="Saves the mesh poses in a .yaml file")
@cloup.option("--gradient_offset", type=float, default=0.0, help="With --save_yaml, adds additionnal measurements points layer for gradient computation with given normal offset")
@cloup.option("--dual", is_flag=True, help="Generates the dual mesh on top of the mesh")
@cloup.option("--similar", is_flag=True, help="Generates similar meshes on top of the mesh (meshes with similar elements nodes but larger resolutions)")
@cloup.option("--similar_number", type=int, default=3, help="Number of similar meshes to generate")
@cloup.option("--output_folder", type=str, default=None, help="Folder where to save the mesh")
@cloup.option("--info", is_flag=True, help="Displays mesh information")
def main(mesh_type, size, resolution, sigma, element_type, save_mesh, save_yaml, gradient_offset, dual, similar, similar_number, output_folder, info):

    if(mesh_type == "sphere"):
        vertices,faces = generateSphericMesh(size, resolution, sigma, element_type, save_mesh, save_yaml, gradient_offset, output_folder)
        if(info):
            getMeshInfo(vertices,faces,element_type)
            plotMesh(vertices,faces,element_type,plotNodes=True,plotEdges=True)
            plt.show()

        #Dual mesh
        if(dual):
            print("Dual mesh are hazardous for now !")
            generateDualSphericMesh(size, resolution, sigma, element_type, save_mesh, save_yaml, gradient_offset, output_folder)
                
        #Similar meshes
        if(similar):
            generateSimilarSphericMeshes(size, resolution, sigma, element_type, similar_number, save_mesh, save_yaml, gradient_offset, output_folder)            

    elif(mesh_type == "circle"):
        vertices,faces = generateCircularMesh(size, resolution, sigma, element_type, save_mesh, save_yaml, gradient_offset, output_folder)
        if(info):
            getMeshInfo(vertices,faces,element_type)
            plotMesh(vertices,faces,element_type,plotEdges=True)
            plt.show()

    else:
        raise ValueError("Invalid mesh type !")
    
if __name__ == "__main__":
    main()