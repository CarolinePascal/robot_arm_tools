#!/usr/bin/env python3.8

## Definition file of the MicrophoneCalibrationServer class
#
# Defines the attributes and methods used to trigger a microphone calibration measurement

import rospy
from std_srvs.srv import Empty,EmptyResponse

import os
import matplotlib.pyplot as plt
import _thread
import time

import sys

import measpy as mp
from measpy.audio import audio_run_measurement

## MicrophoneCalibrationServer
#
# Defines the attributes and methods used to trigger a microphone calibration measurement
class MicrophoneCalibrationServer :
    
    ## Constructor
    def __init__(self):

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

        ## Storage folder name
        self.measurementServerStorageFolder = rospy.get_param("measurementServerStorageFolder")
        try:
            os.mkdir(self.measurementServerStorageFolder)
            rospy.loginfo("Creating " + self.measurementServerStorageFolder + " ...")
        except OSError:
            rospy.logwarn(self.measurementServerStorageFolder + "already exists : its contents will be overwritten !")
            pass 

        ## ROS Service Server used to trigger the calibration measurement
        self.microCalibrationServer = rospy.Service(rospy.get_param("measurementServerName"),Empty,self.measure)

        ## Measurement counter
        self.measurementCounter = 0

    ## Thread safe plotting method for calibration measurements display
    #  @param fig A matplotlib figure
    #  @param ax A matplotlib axes
    def plotting_thread(self,fig,ax):
        while(True):
            time.sleep(2)
            ax.clear()
            self.M0.data['Out4'].plot(ax=ax)
            self.M0.data['In1'].plot(ax=ax)
            fig.canvas.draw_idle()

    ## Method triggering a microphone calibration measurement
    #  @param req An empty ROS service request
    def measure(self, req):
        isGainOK = False
        print("Running micro gain setting test...")

        #Creating figure and axes for calibration measurements display
        #fig = plt.figure()
        #ax = fig.add_subplot(111)

        #Measurement loop
        while(not isGainOK):
            self.measurementCounter += 1

            #Run measurement
            audio_run_measurement(self.M0)
            self.M0.to_csvwav(self.measurementServerStorageFolder+"measurement_"+str(self.measurementCounter))

            #Dispay measurements
            #_thread.start_new_thread(self.plotting_thread,(fig,ax))
            #plt.show()

            #Repeat while microphone calibration is not acceptable
            a = ""
            try:
                a = input("Run the test again ? y/n (default is yes)")
            except SyntaxError: #default
                a = 'y'

            if a == 'n':
                isGainOK = True

            #plt.close("all")

        return EmptyResponse()

def main():
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

if __name__ == "__main__":
    main()
