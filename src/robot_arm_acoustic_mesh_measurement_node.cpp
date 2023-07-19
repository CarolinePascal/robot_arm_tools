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
    RobotVisualTools robotVisualTools;
    
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

    double meshSize, objectSize;
    std::string meshType;
    if(!n.getParam("objectSize",objectSize))
    {
        ROS_ERROR("Unable to retrieve studied object size !");
        throw std::runtime_error("MISSING PARAMETER");
    }
    if(!n.getParam("meshSize",meshSize))
    {
        ROS_ERROR("Unable to retrieve mesh size !");
        throw std::runtime_error("MISSING PARAMETER");
    }
    if(!n.getParam("meshType",meshType))
    {
        ROS_ERROR("Unable to retrieve mesh type !");
        throw std::runtime_error("MISSING PARAMETER");
    }

    if(objectSize > meshSize)
    {
        ROS_WARN("Mesh size below object size : some poses might not be reachable !");
    }

    if(meshType=="sphere")
    {
        robotVisualTools.addSphere("meshSphere",objectPose,meshSize/2,true);
    }

    //Create measurement waypoints poses
    std::vector<geometry_msgs::Pose> waypoints;

    //Get mesh file path, or generate mesh and get its file path
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
            trajectoryFromFile(meshPath, waypoints);
            break;
        }
        catch(const std::exception& e)
        {
            ros::WallDuration(2.0).sleep();
        }
    }
    
    translateTrajectory(waypoints, objectPose.position.x, objectPose.position.y, objectPose.position.z);

    //Main loop
    robot.runMeasurementRoutine(waypoints,false,true,-1);

    //Shut down ROS node 
    ros::shutdown();
    return 0;
}
