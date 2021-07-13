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
    ros::init(argc, argv, "robot_arm_acoustic_directivity_node");  
    ros::AsyncSpinner spinner(0);
    spinner.start();
    ros::WallDuration(1.0).sleep();

    //Robot initialisation TODO More generic approach
    Robot robot;

    //Robot visual tools initialisation
    RobotVisualTools visualTools;

    //Move the robot to its initial configuration
    visualTools.setupUME();
    robot.init();

    //Get the measurement reference pose
    std::vector<double> poseReference;
    ros::NodeHandle n;
    if(!n.getParam("poseReference",poseReference))
    {
        ROS_ERROR("Unable to retrieve measurements reference pose !");
        throw std::runtime_error("MISSING PARAMETER");
    }

    geometry_msgs::Pose centerPose;
    centerPose.position.x = poseReference[0];
    centerPose.position.y = poseReference[1];
    centerPose.position.z = poseReference[2];
    
    tf2::Quaternion offsetQuaternion;
    offsetQuaternion.setRPY(0.0,0.0,M_PI/2 - poseReference[5]);

    //Create measurement waypoints poses
    int N=18;   //10° spaced measurements
    std::vector<geometry_msgs::Pose> waypoints;

    sphericInclinationTrajectory(centerPose,0.0,M_PI/2,0,M_PI,N,waypoints,false,0.0);

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
    robot.runMeasurementRountine(waypoints,measurementServerName,false,storageFolderName+"Positions.csv");

    //Shut down ROS node
    robot.init();   
    ros::waitForShutdown();
    return 0;
}