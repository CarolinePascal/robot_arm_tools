#!/usr/bin/env python3.8

## Definition file of the MicrophoneCalibrationServer class
#
# Defines the attributes and methods used to trigger a microphone calibration measurement

import rospy

import measpy as mp
from measpy.audio import audio_run_measurement
import subprocess

from robot_arm_tools import MeasurementServer

## MicrophoneCalibrationServer
#
# Defines the attributes and methods used to trigger a microphone calibration measurement
class MicrophoneCalibrationServer(MeasurementServer) :
    
    ## Constructor
    def __init__(self):
        super(MicrophoneCalibrationServer,self).__init__()

        self.M0 = mp.Measurement(out_sig='noise',
                                fs=48000,
                                out_sig_freqs=[20,20000],
                                out_map=[1],
                                out_desc=['Sent signal'],
                                out_dbfs=[1.5552],
                                in_map=[1,2,3],
                                out_amp=1.0,
                                in_desc=['Pressure','Voltage','Current'],
                                in_cal=[1.0,1.0,1.0],
                                in_unit=['Pa','V','A'],
                                in_dbfs=[1.7108,1.7120,1.7108],
                                extrat=[0,0],
                                out_sig_fades=[0,0],
                                dur=10,
                                io_sync=0,
                                in_device=4,
                                out_device=4)

    ## Method triggering ALSA drivers recovery
    def recovery(self):

        #sudo chmod -R a+rw /var/run/alsa/
        subprocess.call("pulseaudio -k && /sbin/alsa force-reload", shell=True)
        return(True)

    ## Method triggering a microphone calibration measurement
    def measure(self):

        #Delay used to avoid sound card Alsa related bugs...
        rospy.sleep(1.0)

        isGainOK = False
        print("Running micro gain setting test...")

        #Measurement loop
        while(not isGainOK):
            self.measurementCounter += 1

            #Run measurement
            audio_run_measurement(self.M0)
            self.M0.to_csvwav(self.measurementServerStorageFolder+"measurement_"+str(self.measurementCounter))

            #Repeat while microphone calibration is not acceptable
            a = ""
            try:
                a = input("Run the test again ? y/n (default is yes)")
            except SyntaxError: #default
                a = 'y'

            if a == 'n':
                isGainOK = True

        return(True)

if __name__ == "__main__":

    #Launch ROS node
    rospy.init_node('microphone_calibration_server')

    #Launch ROS service
    MicrophoneCalibrationServer()

    while not rospy.is_shutdown():
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("Shutting down ROS microphone calibration sever")
            break
