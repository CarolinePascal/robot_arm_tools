#include <ros/ros.h>
#include <robot_arm_tools/Robot.h>
#include <robot_arm_tools/RobotTrajectories.h>

#include "robot_arm_acoustic/CreateMesh.h"
#include "AnechoicRoomSupportSetup.hpp"

#include <cmath>

int main(int argc, char **argv)
{
    //ROS node initialisation
    ros::init(argc, argv, "robot_arm_acoustic_mesh_measurement_node");  
    ros::AsyncSpinner spinner(0);
    spinner.start();
    ros::WallDuration(1.0).sleep();

    //Robot initialisation
    Robot robot;
    
    //Get studied object pose
    ros::NodeHandle n;
    std::vector<double> objectPoseArray;
    
    if(!n.getParam("objectPose",objectPoseArray))
    {
        ROS_ERROR("Unable to retrieve studied object pose !");
        throw std::runtime_error("MISSING PARAMETER");
    }

    tf2::Quaternion quaternion;
    quaternion.setRPY(objectPoseArray[3],objectPoseArray[4],objectPoseArray[5]);

    geometry_msgs::Pose objectPose;
    objectPose.position.x = objectPoseArray[0];
    objectPose.position.y = objectPoseArray[1];
    objectPose.position.z = objectPoseArray[2];
    objectPose.orientation =  tf2::toMsg(quaternion);

    addTopSupport(robot,objectPose);

    //Create measurement waypoints poses
    std::vector<geometry_msgs::Pose> waypoints;

    //Get mesh file path, or generate mesh and get its file path
    std::string meshPath;
    try
    {
        //Custom mode
        if(!n.getParam("meshPath",meshPath))
        {
            ROS_ERROR("Unable to retrieve mesh path !");
            throw std::runtime_error("MISSING PARAMETER");
        }
        ros::Duration(10.0).sleep();
    }
    catch(const std::exception& e)
    {
        //Automatic mode
        double objectSize;
        if(!n.getParam("objectSize",objectSize))
        {
            ROS_ERROR("Unable to retrieve studied object size !");
            throw std::runtime_error("MISSING PARAMETER");
        }

        ros::ServiceClient client = n.serviceClient<robot_arm_acoustic::CreateMesh>("/mesh_creation_server");
        robot_arm_acoustic::CreateMesh srv;
        srv.request.type = "sphere";
        srv.request.size = round(10000 * objectSize + 0.05)/10000;
        srv.request.resolution = 0.01;
        if (client.call(srv))
        {
            ROS_INFO("Mesh successfully generated !");
            meshPath = srv.response.mesh_path;
        }
        else
        {
            ROS_ERROR("Failed to generate mesh !");
            throw std::runtime_error("SERVICE FAILURE");
        }
    }
    
    trajectoryFromFile(meshPath, waypoints);
    translateTrajectory(waypoints, objectPose.position.x, objectPose.position.y, objectPose.position.z);

    //Main loop
    robot.runMeasurementRountine(waypoints,false,true);

    //Shut down ROS node
    robot.init();   
    ros::waitForShutdown();
    return 0;
}
