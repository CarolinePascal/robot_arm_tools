#!/usr/bin/env python3

### Microphone gain calibration ###

import rospy
from std_srvs.srv import Empty,EmptyResponse

import os
import matplotlib.pyplot as plt
import _thread
import time

import measpy as mp
from measpy.audio import audio_run_measurement

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
                extrat=[0.0,0.0],
                out_sig_fades=[0.0,0.0],
                dur=5)

        ### ROS Service Server

        self.microCalibrationServer = rospy.Service("/microphone_calibration_server",Empty,self.measure)

    def plotting_thread(self,fig,ax):
        while(True):
            time.sleep(2)
            ax.clear()
            self.M0.data['Out4'].plot(ax=ax)
            self.M0.data['In1'].plot(ax=ax)
            fig.canvas.draw_idle()

    def measure(self, req):

        isGainOK = False
        print("Running micro gain setting test...")

        fig = plt.figure()
        ax = fig.add_subplot(111)

        while(not isGainOK):
            audio_run_measurement(self.M0)

            _thread.start_new_thread(self.plotting_thread,(fig,ax))
            plt.show()

            a = ""
            try:
                a = input("Run the test again ? y/n (default is yes)")
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
            print("Shutting down ROS microphone calibration sever")
            break

if __name__ == "__main__":
    main()