#!/usr/bin/env python3

### Propeller measurements ###

import rospy
from std_srvs.srv import Empty,EmptyResponse

import actionlib
import robot_arm_acoustic.msg

import os

import measpy as mp
from measpy.audio import audio_run_measurement

class PropellerMeasurementServer :
    
    def __init__(self):
        ### Measurements

        self.M1 = mp.Measurement(in_map=[1],
                    in_desc=['In1'],
                    in_cal=[1.0],
                    in_unit=['Pa'],
                    in_dbfs=[1.0/0.593],
                    extrat=[0.0,0.0],
                    out_sig_fades=[0.0,0.0],
                    dur=5)

        ### Data 

        #Storage folder
        self.storageFolder = "/tmp/PropellerMeasurements/"
        try:
            os.mkdir(self.storageFolder)
            rospy.loginfo("The folder " + self.storageFolder + "was created")
        except OSError:
            rospy.logwarn("The folder " + self.storageFolder + "already exists : its contents will be ereased !")
            pass 

        ### ROS Service Server

        self.microServer = rospy.Service("/propeller_measurement_server",Empty,self.measure)
        self.measureCounter = 0

        ### ROS Action Client

        self.propellerActionClient = actionlib.SimpleActionClient("propeller_action", robot_arm_acoustic.msg.PropellerAction)
        self.propellerActionClient.wait_for_server()

    def measure(self, req):
        self.measureCounter += 1

        rospy.sleep(2.0)

        goal = robot_arm_acoustic.msg.PropellerGoal(minimumPower = 1100, maximumPower = 2000, duration = 5, samples = 100)
        self.propellerActionClient.send_goal(goal)
        self.propellerActionClient.wait_for_result()

        audio_run_measurement(self.M1)
        #self.M1.plot()
        #plt.show()
        self.M1.to_csvwav(self.storageFolder+"noise_measurement"+str(self.measureCounter))

        return EmptyResponse()

def main():
    rospy.init_node('propeller_measurement_server')

    PropellerMeasurementServer()

    while not rospy.is_shutdown():
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("Shutting down ROS propeller measurement server")
            break

if __name__ == "__main__":
    main()