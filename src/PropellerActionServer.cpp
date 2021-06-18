#include "robot_arm_acoustic/PropellerActionServer.h"

PropellerActionServer::PropellerActionServer(std::string propellerServerName):m_propellerActionServer(m_nodeHandle, propellerServerName, boost::bind(&PropellerActionServer::sendCommand, this, _1), false), m_propellerServerName(propellerServerName), m_propellerCommand(SERVO_MIN)
{
  //Starting propeller action server
  m_propellerActionServer.start();

  //Retrieving UDP server parameters (propeller)
  std::string IPAddress;
  int port;
  m_nodeHandle.getParam("IPAddress",IPAddress);
  m_nodeHandle.getParam("port",port);

  ROS_INFO("Launching propeller UDP client...");

  //Creating an UDP socket
  if ((m_UDPClient = socket(AF_INET, SOCK_DGRAM, 0)) < 0) 
  {
    perror("UDP socket creation error...\n");
    exit(-1);
  }

  //Setting UDP server socket address
  m_UDPServerAddress.sin_family = AF_INET;
  m_UDPServerAddress.sin_port = htons(port);
  m_UDPServerAddress.sin_addr.s_addr = inet_addr(IPAddress.c_str());

  ROS_INFO("Propeller UDP client launched and ready to send messages !");
}
 
PropellerActionServer::~PropellerActionServer()
{
  close(m_UDPClient);
}
 
void PropellerActionServer::sendCommand(const robot_arm_acoustic::PropellerGoalConstPtr &goal)
{
  bool success = true;
  std::string stringCommand;

  ROS_INFO("%i",goal->samples);

  int deltaT = (int)(goal->duration/goal->samples);
  int deltaPower = (int)((goal->maximumPower - goal->minimumPower)/goal->samples);

  //Sending the propeller command(s)
  for(int i = 0; i < goal->samples; i++)
  {
    //Checking that preempt has not been requested by the client
    if (m_propellerActionServer.isPreemptRequested() || !ros::ok())
    {
      ROS_INFO("%s: Preempted", m_propellerServerName.c_str());
      m_propellerActionServer.setPreempted();
      success = false;
      break;
    }

    //Computing new propeller command
    m_propellerCommand = SERVO_MIN + i*deltaPower;

    if(m_propellerCommand > SERVO_MAX)
    {
      m_propellerCommand = SERVO_MAX;
    }
    else if(m_propellerCommand < SERVO_MIN)
    {
      m_propellerCommand = SERVO_MIN;
    }

    //Sending new propeller command
    stringCommand = std::to_string(m_propellerCommand);
    if(sendto(m_UDPClient, stringCommand.c_str(), strlen(stringCommand.c_str()), 0, (struct sockaddr*)&m_UDPServerAddress, sizeof(m_UDPServerAddress)) < 0) 
    {
      perror("sending error...\n");
      close(m_UDPClient);
      exit(-1);
    }

    ros::WallDuration(deltaT).sleep();
  }

  if(success)
  {
    //Setting the action state to succeeded
    m_result.success = success;
    m_propellerActionServer.setSucceeded(m_result);
  }
}


int main(int argc, char** argv)
{
  ros::init(argc, argv, "propeller_action_server");
 
  PropellerActionServer propeller("propeller_action");
 
  ros::spin();
  return 0;
}


