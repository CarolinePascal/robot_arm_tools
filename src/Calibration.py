#!/usr/bin/env python

### Micro gain calibration processing ###

import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import psd, csd
from scipy.io.wavfile import write

import rospy
from std_srvs.srv import Empty,EmptyResponse

import easyaudio.easyrecord as er
import easyaudio.easysignal as es

import os

class MicroCalibrationProcessing :
    
    def __init__(self):
        ### Measurements

        self.M0 = er.Measure(dur=5)

        ### ROS Service Server

        self.microCalibrationServer = rospy.Service("/micro_calibration_server",Empty,self.task)
        self.measureCounter = 0

    def task(self, req):

        gainOK = False
        print("Running micro gain setting test...")

        while(not gainOK):
            self.M0.runMeasure()
            H0, f = es.tfe(self.M0.x,self.M0.y[:,0],Fs=self.M0.fs,NFFT=16384)
            t = np.linspace(0,self.M0.dur,self.M0.dur*self.M0.fs)
            
            plt.figure(1)
            plt.plot(t,self.M0.x)
            plt.plot(t,self.M0.y[:,0])

            plt.figure(2)
            plt.plot(f,20*np.log10(H0))
            plt.show(block=False)

            a = ""
            try:
                a = raw_input("Run the test again ? y/n (default is yes)")
            except SyntaxError: #default
                a = 'y'

            if a == 'n':
                gainOK = True

            plt.close('all')

        return EmptyResponse()

def main():
    rospy.init_node('MicroCalibrationService')

    MicroCalibrationProcessing()

    while not rospy.is_shutdown():
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("Shutting down ROS MicroCalibrationProcessing node")
            break

if __name__ == "__main__":
    main()