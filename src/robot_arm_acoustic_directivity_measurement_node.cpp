#include <ros/ros.h>
#include <robot_arm_tools/Robot.h>
#include <robot_arm_tools/RobotTrajectories.h>

#include <Eigen/Geometry> 
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

int main(int argc, char **argv)
{
    //ROS node initialisation
    ros::init(argc, argv, "robot_arm_acoustic_directivity_node");  
    ros::AsyncSpinner spinner(0);
    spinner.start();
    ros::WallDuration(1.0).sleep();

    //Robot initialisation
    Robot robot;

    //Get trajectory parameters
    ros::NodeHandle n;
    std::vector<double> trajectoryAxis, centerPoseArray;
    double trajectoryRadius;
    int trajectoryStepsNumber;
   
    if(!n.getParam("trajectoryRadius",trajectoryRadius))
    {
        ROS_ERROR("Unable to retrieve measured trajectory radius !");
        throw std::runtime_error("MISSING PARAMETER");
    }

    if(!n.getParam("trajectoryStepsNumber",trajectoryStepsNumber))
    {
        ROS_ERROR("Unable to retrieve trajectory steps number !");
        throw std::runtime_error("MISSING PARAMETER");
    }

    if(!n.getParam("objectPose",centerPoseArray))
    {
        ROS_ERROR("Unable to retrieve trajectory center pose !");
        throw std::runtime_error("MISSING PARAMETER");
    }

    geometry_msgs::Pose centerPose;
    tf2::Quaternion quaternion;
    quaternion.setRPY(centerPoseArray[3],centerPoseArray[4],centerPoseArray[5]);
    centerPose.position.x = centerPoseArray[0];
    centerPose.position.y = centerPoseArray[1];
    centerPose.position.z = centerPoseArray[2];
    centerPose.orientation =  tf2::toMsg(quaternion);

    //Create measurement waypoints poses
    std::vector<geometry_msgs::Pose> waypoints;

    //Default z=1 trajectory
    //sphericInclinationTrajectory(centerPose,trajectoryRadius,M_PI/2,0,2*M_PI,trajectoryStepsNumber,waypoints);
    sphericAzimuthTrajectory(centerPose, trajectoryRadius, 0, 0, 2*M_PI, trajectoryStepsNumber, waypoints);

    //TODO FIX
    //Eigen::Vector3d RPY = Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d(trajectoryAxis.data()), Eigen::Vector3d(0,0,1)).toRotationMatrix().eulerAngles(0, 1, 2);
    //rotateTrajectory(waypoints, geometry_msgs::Point(centerPose.position.x,centerPose.position.y,centerPose.position.z), RPY[0],RPY[1],RPY[2]);

    //Main loop 
    robot.runMeasurementRoutine(waypoints,false,true,-1);

    //Shut down ROS node   
    ros::shutdown();
    return 0;
}
