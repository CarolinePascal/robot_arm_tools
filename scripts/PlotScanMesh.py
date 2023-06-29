#!/usr/bin/env python3.8

#Utility package
import yaml
import numpy as np

#Mesh package
import meshio

#Point cloud package
import open3d as o3d

#Custom package
from MeshTools import plotMesh

#Fancy plot parameter
import matplotlib.pyplot as plt
plt.rc('font', **{'size': 12, 'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)

if __name__ == "__main__":
    
    import sys

    meshPath = ""
    try:
        meshPath = sys.argv[1]
    except:
        raise ValueError("INVALID MESH PATH !")

    pointCloudPath = ""
    try:
        pointCloudPath = sys.argv[2]
    except:
        pass

    objectPose = np.zeros(3)
    figure = plt.figure()
    axes = figure.add_subplot(projection='3d') 

    if(pointCloudPath != ""):
        pointCloud = o3d.io.read_point_cloud(pointCloudPath) 
        pointCloud = pointCloud.voxel_down_sample(voxel_size=0.005)
        pointCloudPoints = np.asarray(pointCloud.points)
        pointCloudColors = np.asarray(pointCloud.colors)

        axes.scatter(*pointCloudPoints.T,c=pointCloudColors,s=2)

        objectPose = np.average(pointCloudPoints,axis=0)
    
    if(meshPath.split(".")[-1] == "yaml"):
        with open(meshPath,'r') as f:
            controlPoints = yaml.load(meshPath)["poses"]

        axes.scatter(*(np.array(controlPoints).resize((-1,3)) + objectPose).T, label = "Measurement points")
        extents = np.array([getattr(axes, 'get_{}lim'.format(dim))() for dim in 'xyz'])
        sz = extents[:,1] - extents[:,0]
        centers = np.mean(extents, axis=1)
        maxsize = max(abs(sz))
        r = maxsize/2
        for ctr, dim in zip(centers, 'xyz'):
            getattr(axes, 'set_{}lim'.format(dim))(ctr - r, ctr + r)
        plt.show()

    elif(meshPath.split(".")[-1] == "mesh"):
        mesh = meshio.read(meshPath)
        axes = plotMesh(mesh.points + objectPose, mesh.cells_dict["triangle"], "", True, False, axes, False)

    else:
        raise ValueError("INVALID MESH PATH !")

    extents = np.array([getattr(axes, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(axes, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

    axes.set_xlabel("x (m)")
    axes.set_ylabel("y (m)")
    axes.set_zlabel("z (m)")

    meshParams = ".".join(meshPath.split("/")[-1].split(".")[:-1]).split("_")
    meshType = meshPath.split("/")[-2]
    axes.set_title("Measurements on a mesh of type " + str(meshType) + "\n Size : " + str(meshParams[0]) + " m - Resolution : " + str(meshParams[1]) + " m")

    plt.show()

    figure.tight_layout()
    figure.savefig("./ScanMesh.pdf",dpi=300,bbox_inches='tight')
