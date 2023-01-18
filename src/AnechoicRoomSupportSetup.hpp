#include <robot_arm_tools/Robot.h>
#include <robot_arm_tools/RobotVisualTools.h>

void addTopSupport(Robot& robot, geometry_msgs::Pose objectPose)
{
    //Initial collision avoidance during simulation
    if(robot.isSimulated())
    {
        //Move the robot away from the studied object
        geometry_msgs::Pose trickPose = objectPose;
        if(objectPose.position.x > objectPose.position.y)
        {
            trickPose.position.y += 0.2;
        }
        else
        {
            trickPose.position.x += 0.2;
        }
        robot.goToTarget(trickPose);
    }

    //Add the support on top of the studied object
    geometry_msgs::Pose supportPose = objectPose;
    supportPose.position.z += 0.5;

    RobotVisualTools visualTools;
    visualTools.addCylinder("objectSupport", supportPose, 0.01, 1.0, false);
}

void addBottomSupport(){}