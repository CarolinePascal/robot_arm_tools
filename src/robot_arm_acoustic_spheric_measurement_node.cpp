#include <ros/ros.h>
#include <robot_arm_tools/Robot.h>
#include <robot_arm_tools/RobotTrajectories.h>

int main(int argc, char **argv)
{
    //ROS node initialisation
    ros::init(argc, argv, "robot_arm_acoustic_spheric_measurement_node");  
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

    //Get trajectory parameters
    double trajectoryRadius;
    int trajectoryStepsNumber;

    if(!n.getParam("trajectoryRadius",trajectoryRadius))
    {
        ROS_ERROR("Unable to retrieve trajectory radius !");
        throw std::runtime_error("MISSING PARAMETER");
    }

    if(!n.getParam("trajectoryStepsNumber",trajectoryStepsNumber))
    {
        ROS_ERROR("Unable to retrieve trajectory steps number !");
        throw std::runtime_error("MISSING PARAMETER");
    }

    //Create measurement waypoints poses
    std::vector<geometry_msgs::Pose> waypoints;

    sphericAzimuthTrajectory(objectPose, trajectoryRadius, M_PI/2, M_PI/2, 3*M_PI/2, trajectoryStepsNumber, waypoints);
    
    //Main loop
    robot.runMeasurementRoutine(waypoints,false,true);

    //Shut down ROS node  
    ros::shutdown();
    return 0;
}
