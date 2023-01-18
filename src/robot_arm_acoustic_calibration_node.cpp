#include <ros/ros.h>
#include <robot_arm_tools/Robot.h>

#include <std_srvs/Empty.h>

#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <yaml-cpp/yaml.h>
#include <fstream>

int main(int argc, char **argv)
{
    //ROS node initialisation
    ros::init(argc, argv, "robot_arm_acoustic_calibration_node");  
    ros::AsyncSpinner spinner(0);
    spinner.start();
    ros::WallDuration(1.0).sleep();

    //Robot initialisation
    Robot robot;

    //Get ROS service calibration server name and create client
    ros::NodeHandle n;
    std::string measurementServerName;
    if(!n.getParam("measurementServerName",measurementServerName))
    {
        ROS_ERROR("Unable to retrieve measurement server name !");
        throw std::runtime_error("MISSING PARAMETER");
    }

    ros::ServiceClient calibrationClient = n.serviceClient<std_srvs::Empty>(measurementServerName);
    calibrationClient.waitForExistence();

    //Tf listenner initialisation
    tf2_ros::Buffer tfBuffer;
    tf2_ros::TransformListener tfListener(tfBuffer);

    //Switch the robot to manual control mode (user action required)
    do 
    {
        ROS_INFO("Switch the robot to manual control mode, and move the tool to its reference pose - Press enter to continue");
    } while (std::cin.get() != '\n');

    //Run the microphone calibration measurement procedure
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

    //Print reference pose
    std::string endEffectorName;
    if(!n.getParam("endEffectorName",endEffectorName))
    {
        ROS_ERROR("Unable to retrieve measurement server name !");
        throw std::runtime_error("MISSING PARAMETER");
    }
    geometry_msgs::Transform transform;

    try
    {
        transform = tfBuffer.lookupTransform("world", endEffectorName, ros::Time(0), ros::Duration(5.0)).transform;
    } 
    catch (tf2::TransformException &ex) 
    {
        throw std::runtime_error("CANNOT RETRIVE SEEKED TRANSFORM !");
    }

    tf2::Quaternion quaternion;
    tf2::fromMsg(transform.rotation,quaternion);
    double roll, pitch, yaw;
    tf2::Matrix3x3(quaternion).getRPY(roll, pitch, yaw);

    ROS_INFO("Reference pose : \n X : %f \n Y : %f \n Z : %f \n RX : %f \n RY : %f \n RZ : %f",transform.translation.x,transform.translation.y,transform.translation.z,roll,pitch,yaw);

    double objectSize, distanceToObject;
    std::cout << "Distance between reference pose and the studied object : " << std::endl;
    std::cin >> distanceToObject;
    std::cout << "Characteristic size of the studied object : " << std::endl;
    std::cin >> objectSize;

    //Save reference pose [legacy]
    std::string yamlFile = ros::package::getPath("robot_arm_acoustic")+"/config/AcquisitionParameters.yaml";
    YAML::Node config;
    try
    {
        config = YAML::LoadFile(yamlFile);
    }
    catch(const std::exception& e)
    {
        std::ofstream {yamlFile};
        config = YAML::LoadFile(yamlFile);
    }

    if (config["referencePose"]) 
    {
        config.remove("referencePose");
    }
    config["referencePose"].push_back(transform.translation.x);
    config["referencePose"].push_back(transform.translation.y);
    config["referencePose"].push_back(transform.translation.z);
    config["referencePose"].push_back(roll);
    config["referencePose"].push_back(pitch);
    config["referencePose"].push_back(yaw);

    std::ofstream fout(yamlFile);   
    fout << config;

    //Create a dedicated environment file for the studied object
    yamlFile = ros::package::getPath("robot_arm_acoustic")+"/environments/StudiedObject.yaml";
    try
    {
        config = YAML::LoadFile(yamlFile);
    }
    catch(const std::exception& e)
    {
        std::ofstream {yamlFile};
        config = YAML::LoadFile(yamlFile);
    }

    //Studied object
    if (config["studiedObject"]) 
    {
        config.remove("studiedObject");
    }
    YAML::Node configObject = config["studiedObject"];
    configObject["type"] = "sphere";
    
    YAML::Node configObjectPose = configObject["pose"];

    //Get reference pose orientation as matrix
    tf2::Matrix3x3 matrix(quaternion);

    configObjectPose["x"] = transform.translation.x + (distanceToObject+objectSize)*matrix[0][2];
    configObjectPose["y"] = transform.translation.y + (distanceToObject+objectSize)*matrix[1][2];
    configObjectPose["z"] = transform.translation.z + (distanceToObject+objectSize)*matrix[2][2];
    configObjectPose["rx"] = 0.0;
    configObjectPose["ry"] = 0.0;
    configObjectPose["rz"] = 0.0;

    YAML::Node configObjectSize = configObject["size"];
    configObjectSize["radius"] = objectSize;

    configObject["collisions"] = true;
    configObject["robot_base_collisions"] = false;

    //Studied object global parameters
    if (config["objectPose"]) 
    {
        config.remove("objectPose");
    }
    config["objectPose"] = distanceToObject;
    config["objectPose"].push_back(transform.translation.x + (distanceToObject+objectSize)*matrix[0][2]);
    config["objectPose"].push_back(transform.translation.y + (distanceToObject+objectSize)*matrix[1][2]);
    config["objectPose"].push_back(transform.translation.z + (distanceToObject+objectSize)*matrix[2][2]);
    config["objectPose"].push_back(0.0);
    config["objectPose"].push_back(0.0);
    config["objectPose"].push_back(0.0);

    if (config["objectSize"]) 
    {
        config.remove("objectSize");
    }
    config["objectSize"] = objectSize;

    fout = std::ofstream(yamlFile);   
    fout << config;

    ros::waitForShutdown();
    return 0;
}

