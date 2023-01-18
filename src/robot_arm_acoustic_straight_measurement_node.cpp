#include <ros/ros.h>
#include <robot_arm_tools/Robot.h>
#include <robot_arm_tools/RobotTrajectories.h>

#include "AnechoicRoomSupportSetup.hpp"

int main(int argc, char **argv)
{
    //ROS node initialisation
    ros::init(argc, argv, "robot_arm_acoustic_straight_measurement_node");  
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
    double trajectoryStepsSize, distanceToObject;
    int trajectoryStepsNumber;

    if(!n.getParam("trajectoryAxis",trajectoryAxis))
    {
        ROS_ERROR("Unable to retrieve measurements trajectory axis !");
        throw std::runtime_error("MISSING PARAMETER");
    }

    if(!n.getParam("trajectoryStepsNumber",trajectoryStepsNumber))
    {
        ROS_ERROR("Unable to retrieve trajectory steps number !");
        throw std::runtime_error("MISSING PARAMETER");
    }

    if(!n.getParam("trajectoryStepsSize",trajectoryStepsSize))
    {
        ROS_ERROR("Unable to retrieve trajectory steps number !");
        throw std::runtime_error("MISSING PARAMETER");
    }

    if(!n.getParam("distanceToObject",distanceToObject))
    {
        ROS_ERROR("Unable to retrieve distance to object !");
        throw std::runtime_error("MISSING PARAMETER");
    }

    //Create measurement waypoints poses
    geometry_msgs::Pose startingPose;
    startingPose.position.x = objectPose.position.x + distanceToObject*trajectoryAxis[0];
    startingPose.position.y = objectPose.position.y + distanceToObject*trajectoryAxis[1];
    startingPose.position.z = objectPose.position.z + distanceToObject*trajectoryAxis[2];

    geometry_msgs::Pose endingPose;
    endingPose.position.x -= trajectoryStepsSize*(trajectoryStepsNumber-1)*trajectoryAxis[0];
    endingPose.position.y -= trajectoryStepsSize*(trajectoryStepsNumber-1)*trajectoryAxis[1];
    endingPose.position.z -= trajectoryStepsSize*(trajectoryStepsNumber-1)*trajectoryAxis[2];

    std::vector<geometry_msgs::Pose> waypoints;
    straightTrajectory(startingPose, endingPose, trajectoryStepsNumber, waypoints);
    
    //Main loop
    robot.runMeasurementRountine(waypoints,false,true);

    //Shut down ROS node  
    ros::waitForShutdown();
    return 0;
}
