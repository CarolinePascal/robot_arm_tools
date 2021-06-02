#!/usr/bin/env python

### Microphone gain calibration ###

import rospy
from std_srvs.srv import Empty,EmptyResponse

import os
import matplotlib.pyplot as plt

import measpy as mp
from measpy.audio import audio_run_measurement, audio_get_devices

class MicrophoneCalibrationServer :
    
    def __init__(self):
        ### Measurements

        self.M0 = mp.Measurement(out_sig='noise',
                out_map=[4],
                out_desc=['Out4'],
                out_dbfs=[1.0],
                in_map=[1],
                in_desc=['In1'],
                in_cal=[1.0],
                in_unit=['Pa'],
                in_dbfs=[1.0],
                extrat=[1.0,2.0],
                dur=2)

        ### ROS Service Server

        self.microCalibrationServer = rospy.Service("/microphone_calibration_server",Empty,self.measure)

    def measure(self, req):

        isGainOK = False
        print("Running micro gain setting test...")

        while(not isGainOK):
            audio_run_measurement(self.M0)
            self.M0.plot()

            a = ""
            try:
                a = raw_input("Run the test again ? y/n (default is yes)")
            except SyntaxError: #default
                a = 'y'

            if a == 'n':
                isGainOK = True

            plt.close("all")

        return EmptyResponse()

def main():
    rospy.init_node('microphone_calibration_server')

    MicrophoneCalibrationServer()

    while not rospy.is_shutdown():
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("Shutting down ROS microphone_calibration_sever node")
            break

if __name__ == "__main__":
    main()