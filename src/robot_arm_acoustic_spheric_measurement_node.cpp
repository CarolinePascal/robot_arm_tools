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
    ros::init(argc, argv, "robot_arm_acoustic_spheric_measurement_node");  
    ros::AsyncSpinner spinner(0);
    spinner.start();
    ros::WallDuration(1.0).sleep();

    //Robot initialisation TODO More generic approach
    Robot robot;

    //Robot visual tools initialisation
    RobotVisualTools visualTools;

    //Move the robot to its initial configuration

    //Get the object radius, pose and the trajectory radius
    std::vector<double> poseReference, poseObject;
    double radiusObject, radiusTrajectory, distanceToObject;

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

    if(!n.getParam("radiusTrajectory",radiusTrajectory))
    {
        ROS_ERROR("Unable to retrieve trajectory radius !");
        throw std::runtime_error("MISSING PARAMETER");
    }

    if(!n.getParam("distanceToObject",distanceToObject))
    {
        ROS_ERROR("Unable to retrieve distance to object !");
        throw std::runtime_error("MISSING PARAMETER");
    }

    geometry_msgs::Pose objectPose;
    objectPose.position.x = poseObject[0];
    objectPose.position.y = poseObject[1];
    objectPose.position.z = poseObject[2] + distanceToObject;
    
    if(radiusObject != 0)
    {
        visualTools.addSphere("collisionSphere", objectPose, radiusObject + 0.025, false);
    }

    //Create spherical scanning waypoints poses
    tf2::Quaternion leftQuaternion, rightQuaternion;
    leftQuaternion.setRPY(M_PI/2,0,atan2(objectPose.position.y,objectPose.position.x));
    rightQuaternion.setRPY(-M_PI/2,0,atan2(objectPose.position.y,objectPose.position.x));

    int N=10;   //Waypoints number
    std::vector<geometry_msgs::Pose> waypoints;

    /*
    //Close field measurements
    double dinclination = M_PI / (N-1);

    for(int i = 0; i < N; i++)
    {
        sphericInclinationTrajectory(objectPose, radiusTrajectory, i*dinclination, 0, 2*M_PI, N*(i > 0 ? 1 : 0), waypoints); //1+round((N-1)*sin(i*dinclination))   
    }

    //Post-processing
    rotateTrajectory(waypoints,objectPose.position,-M_PI/2,0,atan2(objectPose.position.y,objectPose.position.x));

    for(int i = 0; i < waypoints.size(); i++)
    {      
        if(waypoints[i].position.x*objectPose.position.y/objectPose.position.x > waypoints[i].position.y)
        {
            waypoints[i].orientation = tf2::toMsg(rightQuaternion); 
        }
        else
        {
            waypoints[i].orientation = tf2::toMsg(leftQuaternion); 
        }
    }
    */

    //Far field measurement
    sphericAzimuthTrajectory(objectPose, radiusTrajectory, M_PI/2, M_PI/2, 3*M_PI/2, N, waypoints);

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
