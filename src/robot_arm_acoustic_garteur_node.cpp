#include <ros/ros.h>
#include <robot_arm_tools/Robot.h>
#include <robot_arm_tools/RobotTrajectories.h>

#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

int main(int argc, char **argv)
{
    //ROS node initialisation
    ros::init(argc, argv, "robot_arm_acoustic_garteur_node");  
    ros::AsyncSpinner spinner(0);
    spinner.start();
    ros::WallDuration(1.0).sleep();

    //Robot initialisation
    Robot robot;
    
    //Retreive garteur trajectory from file
    ros::NodeHandle nh;
    ros::NodeHandle nhPrivate("~");
    std::string filePath;
    nhPrivate.getParam("trajectoryFilePath",filePath);

    std::vector<geometry_msgs::Pose> waypoints;
    trajectoryFromFile(filePath, waypoints);

    //Retreive object pose
    std::vector<double> objectPoseArray;
    
    if(!nh.getParam("objectPose",objectPoseArray))
    {
        ROS_ERROR("Unable to retrieve studied object pose !");
        throw std::runtime_error("MISSING PARAMETER");
    }

    geometry_msgs::Point centerPoint;
    centerPoint.x = 0.0;
    centerPoint.y = 0.0;
    centerPoint.z = 0.0;

    rotateTrajectory(waypoints, centerPoint, objectPoseArray[3], objectPoseArray[4], objectPoseArray[5]); 
    translateTrajectory(waypoints, objectPoseArray[0], objectPoseArray[1], objectPoseArray[2]);

    //Main loop
    robot.runMeasurementRoutine(waypoints,false,true,-1,true);

    ros::shutdown();
    return 0;
}

