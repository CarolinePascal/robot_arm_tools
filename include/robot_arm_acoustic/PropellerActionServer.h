/**
 * \file PropellerActionServer.h
 * \brief Header file of the propeller ROS action server class
 *
 * Header file of the propeller ROS action server class - Defines the attributes and methods used to send non-blocking commands to the propeller to perform sound measurements
 *
 */

#include <ros/ros.h>
#include <actionlib/server/simple_action_server.h>
#include "robot_arm_acoustic/PropellerAction.h"

#include <iostream>
#include <fstream>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

#define SERVO_MIN 1100 /*uS*/
#define SERVO_MAX 2000 /*uS*/

/*! \class PropellerActionServer
* \brief Propeller ROS action server class
*/
class PropellerActionServer
{
  public:
 
    /*!
     *  \brief Constructor
     *  \param propellerServerName The name of the propeller ROS action server
     */
    PropellerActionServer(std::string propellerServerName);
  
    /*!
    *  \brief Destructor
    */
    ~PropellerActionServer();
 
    /*!
     *  \brief Sends a non blocking command to the propeller
     *  \param goal
     */
    void sendCommand(const robot_arm_acoustic::PropellerGoalConstPtr &goal);

  private:
 
    ros::NodeHandle m_nodeHandle; /*!< ROS node handle */
    actionlib::SimpleActionServer<robot_arm_acoustic::PropellerAction> m_propellerActionServer; /*!< Propeller ROS action server */
    std::string m_propellerServerName; /*!< Propeller ROS action server name */
    robot_arm_acoustic::PropellerResult m_result; /*!< Propeller ROS action result */

    int m_propellerCommand; /*!< Current propeller command */

    int m_UDPClient;  /*!< UDP client ID */
    struct sockaddr_in m_UDPServerAddress;  /*!< UDP server IP address (propeller) */
};


