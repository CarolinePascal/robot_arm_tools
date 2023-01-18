#!/usr/bin/env python3.8

## Definition file of the MeshCreationServer class
#
# Defines the attributes and methods used to create a measurements mesh

import rospy
from robot_arm_acoustic.srv import CreateMesh, CreateMeshResponse

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/scripts")
from MeshTools import generateSphericMesh

## MeshCreationServer
#
# Defines the attributes and methods used to create a measurements mesh
class MeshCreationServer :
    
    ## Constructor
    def __init__(self):

        ## ROS Service Server
        self.meshCreationServer = rospy.Service("/mesh_creation_server",CreateMesh,self.createMesh)

    ## Method creating a measurements mesh
    #  @param req A CreateMesh containing the mesh parameters
    def createMesh(self, req):
        if(req.type != "sphere"):
            rospy.logerr("Requested mesh type is not yet implemented !")
            raise NotImplementedError("INVALID MESH TYPE")

        generateSphericMesh(req.size, req.resolution, elementType = "P0", saveMesh = True, saveYAML = True)      

        return CreateMeshResponse(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/config/meshes/sphere/S_" + str(req.size) + "_" + str(req.resolution) + ".yaml")

def main():
    #Launch ROS node
    rospy.init_node('mesh_creation_server')

    #Launch ROS service
    MeshCreationServer()

    while not rospy.is_shutdown():
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("Shutting down ROS mesh creation server")
            break

if __name__ == "__main__":
    main()
