import numpy as np
import meshio as meshio
import os

from MeshTools import generateSphericMesh, generateCircularMesh, getMeshInfo, plotMesh

## Function creating a spheric mesh using an icosahedric approximation
#  @param size Size of the sphere as its diameter
#  @param resolution Target resolution of the mesh
#  @param sigma Mesh vertices position standard deviation
#  @param elementType Type of element for triangular faces
#  @param saveMesh Wether to save mesh file or not
#  @param saveYAML Wether to save mesh poses in YAML file or not
def generateSphericMeshUncertainty(size, resolution, sigma, elementType = "P0", saveMesh = False, saveYAML = False):

    try:
        os.makedirs(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/config/meshes/sphere_uncertainty/" + elementType + "/")
    except:
        pass

    vertices,faces = generateSphericMesh(size,resolution,elementType,False,False)
    delta = np.random.normal(0,sigma,vertices.shape)
    vertices += delta

    if(saveMesh):
        meshPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/config/meshes/sphere_uncertainty/" + elementType + "/" + str(size) + "_" + str(resolution) + "_" + str(sigma) + ".mesh"
        print("Saving mesh at " + meshPath)
        meshio.write_points_cells(meshPath, list(vertices), [("triangle",list(faces))])

    if(saveYAML):
        raise NotImplementedError

    return(vertices,faces) 

## Function creating a circular mesh
#  @param size Size of the circle as its diameter
#  @param resolution Target resolution of the mesh
#  @param sigma Mesh vertices position standard deviation
#  @param elementType Type of element for lines
#  @param saveMesh Wether to save mesh file or not
#  @param saveYAML Wether to save mesh poses in YAML file or not
#  @return points, faces Generated mesh points (vertices) and lines (cells)
def generateCircularMeshUncertainty(size, resolution, sigma, elementType = "P1", saveMesh = False, saveYAML = False):

    try:
        os.makedirs(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/config/meshes/circle_uncertainty/" + elementType + "/")
    except:
        pass

    vertices,lines = generateCircularMesh(size,resolution,elementType,False,False)
    delta = np.random.normal(0,sigma,vertices.shape)
    vertices += delta

    if(saveMesh):
        meshPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/config/meshes/circle_uncertainty/" + elementType + "/" + str(size) + "_" + str(resolution) + + "_" + str(sigma) + ".mesh"
        print("Saving mesh at " + meshPath)
        meshio.write_points_cells(meshPath, list(vertices), [("line",lines)])

    if(saveYAML):
        raise NotImplemented

    return(vertices, lines)

if __name__ == "__main__":
    import sys

    meshType = "sphere"
    size = 0.1
    resolution = 0.01
    sigma = 0.005
    elementType = "P0"
    saveMesh = 1
    saveYAML = 0
    info = 0

    try:
        meshType = sys.argv[1]
        size = float(sys.argv[2])
        resolution = float(sys.argv[3])
        sigma = float(sys.argv[4])
        elementType = sys.argv[5]
        saveMesh = int(sys.argv[6])
        saveYAML = int(sys.argv[7])
        info = int(sys.argv[8])
    except:
        print("Invalid size and resolution, switching to default : ")
        print("mesh type = " + meshType)
        print("size = " + str(size))
        print("resolution = " + str(resolution))
        print("element type = " + elementType)

    if(meshType == "sphere"):
        if((saveMesh and not os.path.isfile(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/config/meshes/" + meshType + "_uncertainty/" + str(size) + "_" + str(resolution) + "_" + str(sigma) + ".mesh")) or info):
            vertices,faces = generateSphericMeshUncertainty(size, resolution, sigma, elementType, saveMesh, saveYAML)
            if(info):
                getMeshInfo(vertices,faces,elementType)
                plotMesh(vertices,faces,elementType)

    elif(meshType == "circle"):
        if((saveMesh and not os.path.isfile(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/config/meshes/" + meshType + "_uncertainty/" + str(size) + "_" + str(resolution) + "_" + str(sigma) + ".mesh")) or info):
            vertices,lines = generateCircularMeshUncertainty(size, resolution, sigma, elementType, saveMesh, saveYAML)
            if(info):
                getMeshInfo(vertices,lines,elementType)
                plotMesh(vertices,lines,elementType)

    else:
        print("Invalid mesh type !")
        sys.exit(-1)