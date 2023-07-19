#!/usr/bin/env python3.8

## Definition file of the PropellerMeasurementServer class
#
# Defines the attributes and methods used to trigger a propeller measurement
#
# Do not forget to set the following environment variables : 
# export ROS_MASTER_URI=http://this_computer_IP_address:11311 
# export ROS_IP=this_computer_IP_address 

import rospy

import threading

import measpy as mp
from measpy.audio import audio_run_measurement
import subprocess

from robot_arm_tools import MeasurementServer

## PropellerMeasurementServer
#
# Defines the attributes and methods used to trigger a propeller measurement
class PropellerMeasurementServer(MeasurementServer) :
    
    ## Constructor
    def __init__(self):

        ## Propeller sound measurement
        self.M1 = mp.Measurement(in_map=[1],
                    in_desc=['In1'],
                    in_cal=[1.0],
                    in_unit=['Pa'],
                    in_dbfs=[1.0/0.593],
                    extrat=[0.0,0.0],
                    out_sig_fades=[0.0,0.0],
                    dur=5+1)

        # ROS Service Client
        rospy.wait_for_service("propeller_server")

    ## Method triggering ALSA drivers recovery
    def recovery(self):

        #sudo chmod -R a+rw /var/run/alsa/
        subprocess.call("pulseaudio -k && /sbin/alsa force-reload", shell=True)
        return(True)

    ## Method triggering a propeller measurement
    def measure(self):

        #Delay used to avoid sound card Alsa related bugs...
        rospy.sleep(2.0)

        #Start the measurement on an independant, non-blocking thread
        thread = threading.Thread(target=audio_run_measurement, args=(self.M1,))
        thread.start()
        rospy.sleep(0.5)

        #Send the starting command to the propeller
        rospy.ServiceProxy("propeller_server", Empty)()
        rospy.sleep(0.5)

        #Wait for the measurement to be over
        thread.join()

        #Save measurement results
        self.M1.to_csvwav(self.measurementServerStorageFolder+"noise_measurement"+str(self.measurementCounter))

        return(True)

if __name__ == "__main__":
    
    #Launch ROS node
    rospy.init_node('propeller_measurement_server')

    #Launch ROS service
    PropellerMeasurementServer()

    while not rospy.is_shutdown():
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("Shutting down ROS propeller measurement server")
            break