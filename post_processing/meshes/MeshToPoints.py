import yaml
import meshio
import os
import glob
import numpy as np

## Function converting a .mesh file to a .yaml file containing its vertices
#  @param meshPath Path to the .mesh file
def GeneratePointFile(meshPath):
    mesh = meshio.read(meshPath)
    with open(os.path.dirname(meshPath) + "/" + os.path.basename(meshPath)[:-5] + ".yaml", mode="w+") as file:
        yaml.dump({"meshPoints":mesh.points.flatten().tolist()},file)

if __name__ == "__main__":

    import sys

    try:
        meshPath = sys.argv[1]
    except:
        meshPathArray = np.array(glob.glob("*.mesh"))
        meshPath = input("Choose mesh : " + str([os.path.basename(path) for path in meshPathArray]))
        meshPath = os.getcwd() + "/" + meshPath

    GeneratePointFile(meshPath)
