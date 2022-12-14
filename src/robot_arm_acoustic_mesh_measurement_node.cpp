#include <ros/ros.h>
#include <robot_arm_tools/Robot.h>
#include <robot_arm_tools/RobotTrajectories.h>
#include <robot_arm_tools/RobotVisualTools.h>

#include <std_srvs/Empty.h>

#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <iostream>
#include <fstream>

#include <cmath>

int main(int argc, char **argv)
{
    //ROS node initialisation
    ros::init(argc, argv, "robot_arm_acoustic_mesh_measurement_node");  
    ros::AsyncSpinner spinner(0);
    spinner.start();
    ros::WallDuration(1.0).sleep();

    //Robot initialisation TODO More generic approach
    Robot robot;
    RobotVisualTools visualTools;

    //Get the object radius, pose and the trajectory radius
    std::vector<double> poseReference, poseObject;
    double radiusObject, distanceToObject;

    ros::NodeHandle n;
    if(!n.getParam("poseReference",poseObject))
    {
        ROS_ERROR("Unable to retrieve measurements reference pose !");
        throw std::runtime_error("MISSING PARAMETER");
    }

    if(!n.getParam("radiusObject",radiusObject))
    {
        ROS_ERROR("Unable to retrieve measured object radius !");
        throw std::runtime_error("MISSING PARAMETER");
    }

    if(!n.getParam("distanceToObject",distanceToObject))
    {
        ROS_ERROR("Unable to retrieve distance to object !");
        throw std::runtime_error("MISSING PARAMETER");
    }

    geometry_msgs::Pose objectPose, supportPose;
    objectPose.position.x = poseObject[0];
    objectPose.position.y = poseObject[1];
    objectPose.position.z = poseObject[2] + distanceToObject;
    supportPose = objectPose;
    supportPose.position.z += 0.5;
    
    if(radiusObject != 0)
    {
        visualTools.addSphere("collisionSphere", objectPose, radiusObject, false);
        visualTools.addCylinder("collisionSupport", supportPose, 0.01, 1.0, false);
    }

    std::vector<geometry_msgs::Pose> waypoints;

    std::string meshPath;
    if(!n.getParam("meshPath",meshPath))
    {
        ROS_ERROR("Unable to retrieve mesh path !");
        throw std::runtime_error("MISSING PARAMETER");
    }

    ros::Duration(10.0).sleep();

    trajectoryFromFile(meshPath, waypoints);
    translateTrajectory(waypoints, objectPose.position.x, objectPose.position.y, objectPose.position.z);

    //Get the measurement server name
    std::string measurementServerName, storageFolderName;
    if(!n.getParam("measurementServerName",measurementServerName))
    {
        ROS_ERROR("Unable to retrieve measurement server name !");
        throw std::runtime_error("MISSING PARAMETER");
    }

    //Get the storage folder name
    if(!n.getParam("storageFolderName",storageFolderName))
    {
        ROS_ERROR("Unable to retrieve positions file name !");
        throw std::runtime_error("MISSING PARAMETER");
    }
    
    //Main loop
    robot.runMeasurementRountine(waypoints,false,true);

    //Shut down ROS node
    robot.init();   
    ros::waitForShutdown();
    return 0;
}
