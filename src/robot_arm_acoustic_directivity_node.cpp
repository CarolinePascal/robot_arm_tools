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

    //Command line arguments handling
    if(argc < 3)
    {
        throw std::invalid_argument("MISSING CMD LINE ARGUMENT FOR robot_arm_acoustic_directivity_node !");
        return(1);
    }

    //Robot initialisation TODO More generic approach
    Robot robot("panda_arm", argv[1], "panda_" + std::string(argv[1]));

    //Robot visual tools initialisation
    RobotVisualTools visualTools;

    //Move the robot to its initial configuration
    robot.init();

    //Load object geometry
    std::vector<double> poseReference;

    ROS_INFO("Getting acquisition parameters");
    
    ros::NodeHandle n;
    n.getParam("poseReference",poseReference);

    geometry_msgs::Pose centerPose;
    centerPose.position.x = poseReference[0];
    centerPose.position.y = poseReference[1];
    centerPose.position.z = poseReference[2];
    
    tf2::Quaternion offsetQuaternion;
    offsetQuaternion.setRPY(0.0,0.0,M_PI/2 - poseReference[5]);

    int N=18;   //Waypoints number
    std::vector<geometry_msgs::Pose> waypoints;

    sphericInclinationTrajectory(centerPose,0.0,M_PI/2,0,M_PI,N,waypoints,false,0.0);

    //Post-processing
    tf2::Quaternion initialQuaternion;
    for(int i = 0; i < waypoints.size(); i++)
    {        
        tf2::fromMsg(waypoints[i].orientation,initialQuaternion);
        waypoints[i].orientation = tf2::toMsg(offsetQuaternion*initialQuaternion);
    }

    robot.runMeasurementRountine(waypoints,argv[2],"/tmp/SoundMeasurements/Positions.csv");

    //Shut down ROS node
    robot.init();   
    ros::waitForShutdown();
    return 0;
}