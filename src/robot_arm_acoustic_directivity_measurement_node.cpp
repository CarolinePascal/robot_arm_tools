#include <ros/ros.h>
#include <robot_arm_tools/Robot.h>
#include <robot_arm_tools/RobotTrajectories.h>

#include "AnechoicRoomSupportSetup.hpp"

int main(int argc, char **argv)
{
    //ROS node initialisation
    ros::init(argc, argv, "robot_arm_acoustic_directivity_node");  
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

    //Get trajectory parameters
    std::vector<double> trajectoryAxis;
    double radiusTrajectory, trajectoryStepsSize;
    int trajectoryStepsNumber;
   
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

    //Create measurement waypoints poses
    std::vector<geometry_msgs::Pose> waypoints;

    sphericInclinationTrajectory(objectPose,radiusTrajectory,M_PI/2,trajectoryAxis[0]*(M_PI/2) + M_PI*trajectoryAxis[1],trajectoryAxis[0]*(-M_PI/2),trajectoryStepsNumber,waypoints);

    //Main loop 
    robot.runMeasurementRountine(waypoints,false,true);

    //Shut down ROS node
    robot.init();   
    ros::waitForShutdown();
    return 0;
}
