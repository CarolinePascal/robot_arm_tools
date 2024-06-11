#!/usr/bin/env python3.8

## Definition file of the PropellerMeasurementServer class
#
# Defines the attributes and methods used to trigger a propeller measurement
#
# Do not forget to set the following environment variables : 
# export ROS_MASTER_URI=http://this_computer_IP_address:11311 
# export ROS_IP=this_computer_IP_address 

import rospy
from std_srvs.srv import Empty,EmptyResponse

import threading
import subprocess

import measpy as mp
from measpy.audio import audio_run_measurement

from robot_arm_tools.MeasurementServer import MeasurementServer

## PropellerMeasurementServer
#
# Defines the attributes and methods used to trigger a propeller measurement
class PropellerMeasurementServer(MeasurementServer) :
    
    ## Constructor
    def __init__(self):
        super().__init__()

        #GLOBAL PARAMETERS
        self.device = 3
        self.fs = 48000

        #INPUT PARAMETERS
        self.dur = 5
        self.in_dbfs = 1.7108
        self.in_cal = 1.0

        ## Propeller sound measurement       
        self.in_sig_pressure = mp.Signal(fs = self.fs,
                                         unit = "Pa",
                                         dbfs = self.in_dbfs,
                                         cal = self.in_cal,
                                         desc = "input_pressure")
        
        self.measurement = mp.Measurement(in_sig = [self.in_sig_pressure],
                                          in_map = [1],
                                          dur = self.dur,
                                          in_device = self.device,
                                          device_type = "audio")
                                          
        # ROS Service Client
        rospy.wait_for_service("propeller_server")

    ## Method triggering ALSA drivers recovery
    def recovery(self):

        #sudo chmod -R a+rw /var/run/alsa/
        subprocess.run("pulseaudio -k && /sbin/alsa force-reload", shell=True)
        return(True)

    ## Method triggering a propeller measurement
    def measure(self):

        #Delay used to avoid sound card Alsa related bugs...
        rospy.sleep(2.0)

        #Start the measurement on an independant, non-blocking thread
        thread = threading.Thread(target=audio_run_measurement, args=(self.measurement,))
        thread.start()
        rospy.sleep(0.5)

        #Send the starting command to the propeller
        rospy.ServiceProxy("propeller_server", Empty)()
        rospy.sleep(0.5)

        #Wait for the measurement to be over
        thread.join()

        #Save measurement results
        self.measurement.to_dir(self.measurementServerStorageFolder+"noise_measurement"+str(self.measurementServerCounter))

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