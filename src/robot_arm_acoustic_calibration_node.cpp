#include <ros/ros.h>
#include <robot_arm_tools/Robot.h>

#include <std_srvs/Empty.h>

#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <yaml-cpp/yaml.h>
#include <fstream>

#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

int main(int argc, char **argv)
{
    //ROS node initialisation
    ros::init(argc, argv, "robot_arm_acoustic_calibration_node");  
    ros::AsyncSpinner spinner(0);
    spinner.start();
    ros::WallDuration(1.0).sleep();

    //Robot initialisation
    Robot robot;

    //Tf listenner initialisation
    tf2_ros::Buffer tfBuffer;
    tf2_ros::TransformListener tfListener(tfBuffer);

    //Switch the robot to manual control mode (user action required)
    do 
    {
        ROS_INFO("Switch the robot to manual control mode, and move the tool to its reference pose - Press enter to continue");
    } while (std::cin.get() != '\n');

    //Print reference pose
    geometry_msgs::Pose referencePose = robot.getCurrentPose();

    tf2::Quaternion quaternion;
    tf2::fromMsg(referencePose.orientation,quaternion);
    double roll, pitch, yaw;
    tf2::Matrix3x3(quaternion).getRPY(roll, pitch, yaw);

    ROS_INFO("Reference pose : \n X : %f \n Y : %f \n Z : %f \n RX : %f \n RY : %f \n RZ : %f",referencePose.position.x,referencePose.position.y,referencePose.position.z,roll,pitch,yaw);

    double objectSize, distanceToObject;
    std::cout << "Distance between the microphone and the propeller center (m) : " << std::endl;
    std::cin >> distanceToObject;
    std::cout << "Characteristic size of the propeller : " << std::endl;
    std::cin >> objectSize;

    //Save reference pose
    std::string yamlFile = ros::package::getPath("robot_arm_acoustic")+"/config/PropellerCalibrationParameters.yaml";
    YAML::Node config;
    try
    {
        config = YAML::LoadFile(yamlFile);
        ROS_WARN("%s already exists, its contents will be overwritten !",yamlFile.c_str());
    }
    catch(const std::exception& e)
    {
        std::ofstream {yamlFile};
        config = YAML::LoadFile(yamlFile);
    }

    if(config["referencePose"]) 
    {
        config.remove("referencePose");
    }
    config["referencePose"].push_back(referencePose.position.x);
    config["referencePose"].push_back(referencePose.position.y);
    config["referencePose"].push_back(referencePose.position.z);
    config["referencePose"].push_back(roll);
    config["referencePose"].push_back(pitch);
    config["referencePose"].push_back(yaw);

    if(config["jointsValues"]) 
    {
        config.remove("jointsValues");
    }
    std::vector<double> jointsValues = robot.getCurrentJointsValues();
    for(std::vector<double>::iterator it = jointsValues.begin(); it != jointsValues.end(); it++)
    {
        config["jointsValues"].push_back(*it);
    }

    std::ofstream fout(yamlFile);   
    fout << config;

    //Create a dedicated environment file for the studied object
    yamlFile = ros::package::getPath("robot_arm_acoustic")+"/config/environments/Propeller.yaml";
    try
    {
        config = YAML::LoadFile(yamlFile);
        ROS_WARN("%s already exists, its contents will be overwritten !",yamlFile.c_str());
    }
    catch(const std::exception& e)
    {
        std::ofstream {yamlFile};
        config = YAML::LoadFile(yamlFile);
    }

    //Studied object
    if(config["propeller"]) 
    {
        config.remove("propeller");
    }
    YAML::Node configObject = config["propeller"];
    configObject["type"] = "cylinder";
    
    YAML::Node configObjectPose = configObject["pose"];

    //Get reference pose orientation as matrix
    tf2::Matrix3x3 matrix(quaternion);

    geometry_msgs::Pose objectPose;
    //objectPose.position.x = referencePose.position.x + distanceToObject*matrix[0][2];
    //objectPose.position.y = referencePose.position.y + distanceToObject*matrix[1][2];
    //objectPose.position.z = referencePose.position.z + distanceToObject*matrix[2][2];
    objectPose.position.x = referencePose.position.x;
    objectPose.position.y = referencePose.position.y;
    objectPose.position.z = referencePose.position.z - distanceToObject;
    objectPose.orientation.w = 1.0;

    configObjectPose["x"] = objectPose.position.x;
    configObjectPose["y"] = objectPose.position.y;
    configObjectPose["z"] = objectPose.position.z;
    configObjectPose["rx"] = 0.0;
    configObjectPose["ry"] = 0.0;
    configObjectPose["rz"] = 0.0;

    YAML::Node configObjectSize = configObject["size"];
    configObjectSize["radius"] = objectSize/2;
    configObjectSize["height"] = distanceToObject*2;

    configObject["collisions"] = true;
    configObject["robot_base_collisions"] = false;

    //Studied object global parameters
    if(config["objectPose"]) 
    {
        config.remove("objectPose");
    }
    //config["objectPose"] = distanceToObject;
    config["objectPose"].push_back(objectPose.position.x);
    config["objectPose"].push_back(objectPose.position.y);
    config["objectPose"].push_back(objectPose.position.z);
    config["objectPose"].push_back(0.0);
    config["objectPose"].push_back(0.0);
    config["objectPose"].push_back(1.5708); //For trajectory orientation !

    if(config["objectSize"]) 
    {
        config.remove("objectSize");
    }
    config["objectSize"] = objectSize;

    //Studied object base
    if(config["propeller_base"]) 
    {
        config.remove("propeller_base");
    }
    YAML::Node configObjectBase = config["propeller_base"];
    configObjectBase["type"] = "cylinder";
    
    YAML::Node configObjectBasePose = configObjectBase["pose"];

    //objectPose.position.x = referencePose.position.x + (2*distanceToObject + 0.5)*matrix[0][2];
    //objectPose.position.y = referencePose.position.y + (2*distanceToObject + 0.5)*matrix[1][2];
    //objectPose.position.z = referencePose.position.z + (2*distanceToObject + 0.5)*matrix[2][2];
    objectPose.position.x = referencePose.position.x;
    objectPose.position.y = referencePose.position.y;
    objectPose.position.z = referencePose.position.z - (2*distanceToObject + 0.5);
    objectPose.orientation.w = 1.0;

    configObjectBasePose["x"] = objectPose.position.x;
    configObjectBasePose["y"] = objectPose.position.y;
    configObjectBasePose["z"] = objectPose.position.z;
    configObjectBasePose["rx"] = 0.0;
    configObjectBasePose["ry"] = 0.0;
    configObjectBasePose["rz"] = 1.5708;    //To set the trajectory orientation regarding to the robot

    YAML::Node configObjectBaseSize = configObjectBase["size"];
    configObjectBaseSize["radius"] = 0.05;
    configObjectBaseSize["height"] = 1.0;

    configObjectBase["collisions"] = true;
    configObjectBase["robot_base_collisions"] = false;


    fout = std::ofstream(yamlFile);   
    fout << config;

    //robot.runMeasurementRoutine(std::vector<geometry_msgs::Pose>(),true,true);

    //Shutdown node
    ros::shutdown();
    return 0;
}

