#!/usr/bin/env python3.8

## Definition file of the SoundMeasurementServer class
#
# Defines the attributes and methods used to trigger a sound measurement

import rospy
from std_srvs.srv import Empty,EmptyResponse

import os
import sys

import measpy as mp
from measpy.audio import audio_run_measurement


## SoundMeasurementServer
#
# Defines the attributes and methods used to trigger a sound measurement
class SoundMeasurementServer :
    
    ## Constructor
    def __init__(self):

        self.M1 = mp.Measurement(out_sig='logsweep',
                                fs=96000,
                                out_sig_freqs=[20,20000],
                                out_map=[1],
                                out_desc=['Sent signal'],
                                out_dbfs=[1.5552],
                                in_map=[1,2,3,4],
                                out_amp=1.0,
                                in_desc=['Pressure','Voltage','Current','Sent signal'],
                                in_cal=[1.0,0.1,0.1,1.0],
                                in_unit=['Pa','V','A','V'],
                                in_dbfs=[1.7108,1.7108,1.7108,1.7108],
                                extrat=[0,0],
                                out_sig_fades=[0,0],
                                dur=10,
                                io_sync=0,
                                in_device=4,
                                out_device=4)

        self.M2 = mp.Measurement(out_sig='noise',
                                fs=96000,
                                out_sig_freqs=[20,20000],
                                out_map=[1],
                                out_desc=['Sent signal'],
                                out_dbfs=[1.5552],
                                in_map=[1,2,3,4],
                                out_amp=1.0,
                                in_desc=['Pressure','Voltage','Current','Sent signal'],
                                in_cal=[1.0,0.1,0.1,1.0],
                                in_unit=['Pa','V','A','V'],
                                in_dbfs=[1.7108,1.7108,1.7108,1.7108],
                                extrat=[0,0],
                                out_sig_fades=[0,0],
                                dur=10,
                                io_sync=0,
                                in_device=4,
                                out_device=4)

        ## Storage folder name
        self.measurementServerStorageFolder = rospy.get_param("measurementServerStorageFolder")
        try:
            os.mkdir(self.measurementServerStorageFolder)
            rospy.loginfo("Creating " + self.measurementServerStorageFolder + " ...")
        except OSError:
            rospy.logwarn(self.measurementServerStorageFolder + "already exists : its contents will be overwritten !")
            pass 

        ## ROS Service Server
        self.microServer = rospy.Service("/sound_measurement_server",Empty,self.measure)

        ## Measurement counter
        self.measurementCounter = 0

    ## Method triggering a sound measurement
    #  @param req An empty ROS service request
    def measure(self, req):
        self.measurementCounter += 1
        
        #Delay used to avoid sound card Alsa related bugs...
        rospy.sleep(1.0)

        #Run measurement
        audio_run_measurement(self.M1,progress=False)

        #[Debug] Plot measurement
        #self.M1.plot()
        #plt.show()
        self.M1.to_csvwav(self.measurementServerStorageFolder+"sweep_measurement_"+str(self.measurementCounter))

        rospy.sleep(3.0)

        audio_run_measurement(self.M2,progress=False)

        self.M2.to_csvwav(self.measurementServerStorageFolder+"noise_measurement_"+str(self.measurementCounter))

        #Delay used to avoid sound card Alsa related bugs...
        #rospy.sleep(0.5)

        return EmptyResponse()

def main():
    #Launch ROS node
    rospy.init_node('sound_measurement_server')

    #Launch ROS service
    SoundMeasurementServer()

    while not rospy.is_shutdown():
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("Shutting down ROS sound measurement server")
            break

if __name__ == "__main__":
    main()
