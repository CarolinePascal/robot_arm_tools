#include <ros/ros.h>
#include <robot_arm_tools/Robot.h>
#include <robot_arm_tools/RobotTrajectories.h>

int main(int argc, char **argv)
{
    //ROS node initialisation
    ros::init(argc, argv, "robot_arm_acoustic_straight_measurement_node");  
    ros::AsyncSpinner spinner(0);
    spinner.start();
    ros::WallDuration(1.0).sleep();

    //Robot initialisation
    Robot robot;

    //Get trajectory parameters
    ros::NodeHandle n;
    std::vector<double> trajectoryAxis, startPoseArray;
    double trajectoryStepSize;
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

    if(!n.getParam("trajectoryStepSize",trajectoryStepSize))
    {
        ROS_ERROR("Unable to retrieve trajectory steps size !");
        throw std::runtime_error("MISSING PARAMETER");
    }

    if(!n.getParam("trajectoryStartPose",startPoseArray))
    {
        ROS_ERROR("Unable to retrieve trajectory start pose !");
        throw std::runtime_error("MISSING PARAMETER");
    }

    geometry_msgs::Pose startPose;
    tf2::Quaternion quaternion;
    quaternion.setRPY(startPoseArray[3],startPoseArray[4],startPoseArray[5]);
    startPose.position.x = startPoseArray[0];
    startPose.position.y = startPoseArray[1];
    startPose.position.z = startPoseArray[2];
    startPose.orientation =  tf2::toMsg(quaternion);

    geometry_msgs::Pose endPose = startPose;
    endPose.position.x += trajectoryStepSize*(trajectoryStepsNumber-1)*trajectoryAxis[0];
    endPose.position.y += trajectoryStepSize*(trajectoryStepsNumber-1)*trajectoryAxis[1];
    endPose.position.z += trajectoryStepSize*(trajectoryStepsNumber-1)*trajectoryAxis[2];

    std::vector<geometry_msgs::Pose> waypoints;
    straightTrajectory(startPose, endPose, trajectoryStepsNumber, waypoints);
    
    //Main loop
    robot.runMeasurementRoutine(waypoints,false,true,M_PI/2);

    //Shut down ROS node  
    ros::shutdown();
    return 0;
}
