#include <ros/ros.h>
#include <robot_arm_tools/Robot.h>
#include <robot_arm_tools/RobotTrajectories.h>
#include <robot_arm_tools/RobotVisualTools.h>

#include <cmath>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

int main(int argc, char **argv)
{
    //ROS node initialisation
    ros::init(argc, argv, "robot_arm_acoustic_mesh_measurement_node");  
    ros::AsyncSpinner spinner(0);
    spinner.start();
    ros::WallDuration(1.0).sleep();

    //Robot initialisation
    Robot robot;

    //Create measurement waypoints poses
    std::vector<geometry_msgs::Pose> waypoints;

    //Get mesh file path, or generate mesh and get its file path
    ros::NodeHandle n;
    std::string meshPath;
    if(!n.getParam("meshPath",meshPath))
    {
        ROS_ERROR("Unable to retrieve mesh path !");
        throw std::runtime_error("MISSING PARAMETER");
    }
    
    double startTime = ros::WallTime::now().toSec();

    while(ros::WallTime::now().toSec() - startTime < 10.0)
    {
        try
        {
            trajectoryFromFile(waypoints, meshPath);
            break;
        }
        catch(const std::exception& e)
        {
            ros::WallDuration(2.0).sleep();
        }
    }
    if(waypoints.empty())
    {
        ROS_ERROR("Unable to retrieve mesh waypoints !");
        throw std::runtime_error("MISSING PARAMETER");
    }


    //Main loop
    robot.runMeasurementRoutine(waypoints,false,true,-1,true,false);

    //Shut down ROS node 
    ros::shutdown();
    return 0;
}
