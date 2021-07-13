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
    visualTools.setupUME();
    robot.init();

    //Get the object radius, pose and the trajectory radius
    std::vector<double> poseObject;
    double radiusObject;
    double radiusTrajectory;

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

    geometry_msgs::Pose centerPose;
    centerPose.position.x = poseObject[0];
    centerPose.position.y = poseObject[1];
    centerPose.position.z = poseObject[2] - 0.04;

    //Define and add collisions objects
    geometry_msgs::Pose boxPose = centerPose;
    boxPose.position.z -= radiusObject + 0.1;
    
    if(radiusObject != 0)
    {
        visualTools.addSphere("collisionSphere", centerPose, radiusObject + 0.025, false);
        visualTools.addBox("collisionBox", boxPose, 0.05, 0.05, 0.2, false);
    }

    //Create spherical scanning waypoints poses
    tf2::Quaternion leftQuaternion, rightQuaternion;
    leftQuaternion.setRPY(M_PI/2,0,atan2(centerPose.position.y,centerPose.position.x));
    rightQuaternion.setRPY(-M_PI/2,0,atan2(centerPose.position.y,centerPose.position.x));

    int N=10;   //Waypoints number
    std::vector<geometry_msgs::Pose> waypoints;

    double dinclination = M_PI / (N-1);

    for(int i = 0; i < N; i++)
    {
        sphericInclinationTrajectory(centerPose, radiusTrajectory, i*dinclination, 0, 2*M_PI, N*(i > 0 ? 1 : 0), waypoints); //1+round((N-1)*sin(i*dinclination))   
    }

    //Post-processing
    rotateTrajectory(waypoints,centerPose.position,-M_PI/2,0,atan2(centerPose.position.y,centerPose.position.x));

    for(int i = 0; i < waypoints.size(); i++)
    {      
        if(waypoints[i].position.x*centerPose.position.y/centerPose.position.x > waypoints[i].position.y)
        {
            waypoints[i].orientation = tf2::toMsg(rightQuaternion); 
        }
        else
        {
            waypoints[i].orientation = tf2::toMsg(leftQuaternion); 
        }
    }

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