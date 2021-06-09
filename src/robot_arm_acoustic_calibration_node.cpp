#include <ros/ros.h>
#include <robot_arm_tools/Robot.h>
#include <robot_arm_tools/RobotTrajectories.h>
#include <robot_arm_tools/RobotVisualTools.h>

#include <std_srvs/Empty.h>

#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <yaml-cpp/yaml.h>

int main(int argc, char **argv)
{
    //ROS node initialisation
    ros::init(argc, argv, "robot_arm_acoustic_calibration_node");  
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

    //Get ROS service calibration server name and create client
    ros::NodeHandle n;
    ros::ServiceClient calibrationClient = n.serviceClient<std_srvs::Empty>(argv[2]);

    //Robot visual tools initialisation
    RobotVisualTools visualTools;

    //Move the robot to its initial configuration
    //robot.init();

    //Tf listenner initialisation
    tf2_ros::Buffer tfBuffer;
    tf2_ros::TransformListener tfListener(tfBuffer);

    //Switch the robot to manual control mode (user action required)
    do 
    {
        ROS_INFO("Switch the robot to manual control mode, and move the tool to its reference position - Press enter to continue");
    } while (std::cin.get() != '\n');

    std_srvs::Empty request;

    if(calibrationClient.call(request))
    {
        ROS_INFO("Calibration - done !");
    }

    else
    {
        throw std::runtime_error("ERROR DURING CALIBRATION !");
        ros::waitForShutdown();
        return 1;
    }

    //Display reference position

    geometry_msgs::Transform transform;
    try
    {
        transform = tfBuffer.lookupTransform("world", std::string("panda_")+std::string(argv[1]), ros::Time(0), ros::Duration(5.0)).transform;
    } 
    catch (tf2::TransformException &ex) 
    {
        throw std::runtime_error("CANNOT RETRIVE SEEKED TRANSFORM !");
    }

    tf2::Quaternion quaternion;
    tf2::fromMsg(transform.rotation,quaternion);
    double roll, pitch, yaw;
    tf2::Matrix3x3(quaternion).getRPY(roll, pitch, yaw);

    ROS_INFO("Reference position : \n X : %f \n Y : %f \n Z : %f \n RX : %f \n RY : %f \n RZ : %f",transform.translation.x,transform.translation.y,transform.translation.z,roll,pitch,yaw);

    //Save reference position
    
    std::string yamlFile = ros::package::getPath("robot_arm_acoustic")+"/config/AcquisitionParameters.yaml";

    YAML::Node config = YAML::LoadFile(yamlFile);

    if (config["poseReference"]) 
    {
        config.remove("poseReference");
        config["poseReference"].push_back(transform.translation.x);
        config["poseReference"].push_back(transform.translation.y);
        config["poseReference"].push_back(transform.translation.z);
        config["poseReference"].push_back(roll);
        config["poseReference"].push_back(pitch);
        config["poseReference"].push_back(yaw);
    }

    ros::waitForShutdown();
    return 0;
}

