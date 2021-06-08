#!/usr/bin/env python

### Micro processing ###

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

class MicroProcessing :
    
    def __init__(self):
        ### Measurements

        self.M1 = er.Measure(dur=10)
        self.M2 = er.Measure(dur=10,outSig='logsweep')

        ### Data 

        #Storage folder
        self.storageFolder = "/tmp/SoundMeasurements/"
        try:
            os.mkdir(self.storageFolder)
            rospy.loginfo("The folder " + self.storageFolder + "was created")
        except OSError:
            rospy.logwarn("The folder " + self.storageFolder + "already exists : its contents will be ereased !")
            pass 

        self.M1.saveGeneratedSound(self.storageFolder+"Measure1GeneratedSound.wav")
        self.M1.saveToCSV(self.storageFolder+"Measure1Parameters.csv")

        self.M2.saveGeneratedSound(self.storageFolder+"Measure2GeneratedSound.wav")
        self.M2.saveToCSV(self.storageFolder+"Measure2Parameters.csv")

        ### ROS Service Server

        self.microServer = rospy.Service("/micro_measurement_server",Empty,self.task)
        self.measureCounter = 0

    def task(self, req):
        self.measureCounter += 1

        #rospy.sleep(2.0)
        #self.M1.runMeasure()
        #self.M1.saveRecordedSound(self.storageFolder+"Measure1RecordedSound"+str(self.measureCounter)+".wav")

        rospy.sleep(2.0)
        self.M2.runMeasure()
        self.M2.saveRecordedSound(self.storageFolder+"Measure2RecordedSound"+str(self.measureCounter)+".wav")

        return EmptyResponse()

def main():
    rospy.init_node('MicroService')

    MicroProcessing()

    while not rospy.is_shutdown():
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("Shutting down ROS MicroService node")
            break

if __name__ == "__main__":
    main()