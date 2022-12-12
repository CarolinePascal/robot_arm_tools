import yaml
import meshio
import os
import glob
import numpy as np
import trimesh

## Function converting a .mesh file to a .yaml file containing its vertices
#  @param meshPath Path to the .mesh file
def GeneratePointFile(meshPath, elementType = 'P1'):
    mesh = meshio.read(meshPath)
    meshPoses = np.empty((0,6))
    Points = None

    if(elementType == 'P0'):
        Points = trimesh.Trimesh(mesh.points,mesh.cells_dict["triangle"])
        
    elif(elementType == 'P1'):
        Points = mesh.points

    for point in Points:
            radius = np.linalg.norm(point)
            inclination, azimuth = np.arccos(point[2]/radius),np.arctan2(point[1],point[0])
            meshPoses = np.vstack((meshPoses,np.hstack((point,[np.pi,inclination,azimuth]))))

    print(meshPoses)

    with open(os.path.dirname(meshPath) + "/" + os.path.basename(meshPath)[:-5] + ".yaml", mode="w+") as file:
        yaml.dump({"elementType":elementType},file)
        yaml.dump({"poses":meshPoses.tolist()},file)

if __name__ == "__main__":

    import sys

    try:
        meshPath = sys.argv[1]
    except:
        meshPathArray = np.array(glob.glob("*.mesh"))
        meshPath = input("Choose mesh : " + str([os.path.basename(path) for path in meshPathArray]))
        meshPath = os.getcwd() + "/" + meshPath

    GeneratePointFile(meshPath)
