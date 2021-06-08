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
    ros::init(argc, argv, "robot_arm_acoustic_measurement_node");  
    ros::AsyncSpinner spinner(0);
    spinner.start();
    ros::WallDuration(1.0).sleep();

    //Command line arguments handling
    if(argc < 3)
    {
        throw std::invalid_argument("MISSING CMD LINE ARGUMENT FOR acoustic_micro_node !");
        return(1);
    }

    //Robot initialisation TODO More generic approach
    Robot robot("panda_arm", argv[1], "panda_" + std::string(argv[1]));

    //Robot visual tools initialisation
    RobotVisualTools visualTools;

    //Move the robot to its initial configuration
    visualTools.setupUME();
    robot.init();

    //Load object geometry
    std::vector<double> poseObject;
    double radiusObject;
    double radiusTrajectory;

    ROS_INFO("Getting acquisition parameters");

    ros::NodeHandle n;
    n.getParam("poseReference",poseObject);
    n.getParam("radiusObject",radiusObject);
    n.getParam("radiusTrajectory",radiusTrajectory);

    geometry_msgs::Pose centerPose;
    centerPose.position.x = poseObject[0];
    centerPose.position.y = poseObject[1];
    centerPose.position.z = poseObject[2] - 0.04;

    std::cout << centerPose << std::endl;

    std::cout<<radiusObject<<std::endl;

    geometry_msgs::Pose boxPose = centerPose;
    boxPose.position.z -= radiusObject + 0.1;
    
    if(radiusObject != 0)
    {
        visualTools.addSphere("collisionSphere", centerPose, radiusObject + 0.025, false);
        visualTools.addBox("collisionBox", boxPose, 0.05, 0.05, 0.2, false);
    }

    //Compute spherical scanning trajectory points
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

    //Check creation
    robot.runMeasurementRountine(waypoints,argv[2],"/tmp/SoundMeasurements/Positions.csv");

    //Shut down ROS node
    robot.init();   
    ros::waitForShutdown();
    return 0;
}