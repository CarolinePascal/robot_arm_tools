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

    //Robot initialisation
    Robot robot;

    //Robot visual tools initialisation
    RobotVisualTools visualTools;

    //Move the robot to its initial configuration
    visualTools.setupUME();

    //Get the object radius, pose and the trajectory radius
    std::vector<double> poseReference, trajectoryAxis;
    double radiusObject, radiusTrajectory, trajectoryStepsSize, distanceToObject;
    int trajectoryStepsNumber;

    ros::NodeHandle n;
    if(!n.getParam("poseReference",poseReference))
    {
        ROS_ERROR("Unable to retrieve measurements reference pose !");
        throw std::runtime_error("MISSING PARAMETER");
    }

    if(!n.getParam("radiusObject",radiusObject))
    {
        ROS_ERROR("Unable to retrieve measured object radius !");
        throw std::runtime_error("MISSING PARAMETER");
    }
    
    if(!n.getParam("radiusTrajectory",radiusTrajectory))
    {
        ROS_ERROR("Unable to retrieve measured trajectory radius !");
        throw std::runtime_error("MISSING PARAMETER");
    }

    if(!n.getParam("trajectoryStepsNumber",trajectoryStepsNumber))
    {
        ROS_ERROR("Unable to retrieve trajectory steps number !");
        throw std::runtime_error("MISSING PARAMETER");
    }

    if(!n.getParam("trajectoryAxis",trajectoryAxis))
    {
        ROS_ERROR("Unable to retrieve trajectory axis !");
        throw std::runtime_error("MISSING PARAMETER");
    }

    if(!n.getParam("distanceToObject",distanceToObject))
    {
        ROS_ERROR("Unable to retrieve distance to object !");
        throw std::runtime_error("MISSING PARAMETER");
    }

    tf2::Quaternion quaternion;
    quaternion.setRPY(poseReference[3],poseReference[4],poseReference[5]);
    tf2::Matrix3x3 matrix(quaternion);

    geometry_msgs::Pose objectPose;
    objectPose.position.x = poseReference[0] + (distanceToObject+radiusObject)*matrix[0][2];
    objectPose.position.y = poseReference[1] + (distanceToObject+radiusObject)*matrix[1][2];
    objectPose.position.z = poseReference[2] + (distanceToObject+radiusObject)*matrix[2][2];
    
    if(radiusObject != 0)
    {
        visualTools.addSphere("collisionSphere", objectPose, radiusObject, false);
        visualTools.addCylinder("collisonCylinder", objectPose, 0.01, 1.0, false);
    }

    //Create measurement waypoints poses
    int N=trajectoryStepsNumber;   //10° spaced measurements
    std::vector<geometry_msgs::Pose> waypoints;

    sphericInclinationTrajectory(objectPose,radiusTrajectory,M_PI/2,trajectoryAxis[0]*(M_PI/2) + M_PI*trajectoryAxis[1],trajectoryAxis[0]*(-M_PI/2),N,waypoints);

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
