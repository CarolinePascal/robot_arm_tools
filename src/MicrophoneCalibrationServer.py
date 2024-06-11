#!/usr/bin/env python3.8

## Definition file of the MicrophoneCalibrationServer class
#
# Defines the attributes and methods used to trigger a microphone calibration measurement

import rospy

import subprocess
from copy import deepcopy

import measpy as mp
from measpy.audio import audio_run_measurement

from robot_arm_tools.MeasurementServer import MeasurementServer

## MicrophoneCalibrationServer
#
# Defines the attributes and methods used to trigger a microphone calibration measurement
class MicrophoneCalibrationServer(MeasurementServer) :
    
    ## Constructor
    def __init__(self):
        super().__init__()

        #GLOBAL PARAMETERS
        self.device = 3
        self.fs = 96000
        self.mono_left = True
        self.mono_right = True

        #OUTPUT PARAMETERS
        self.dur = 15
        self.amp = 0.5
        self.freq_min = 1
        self.freq_max = 30000
        self.out_dbfs = 1.565 #1.5552
        self.added_time = 0.5

        #INPUT PARAMETERS
        self.in_dbfs = 1.7108
        self.in_cal = 0.0502

        #SIGNALS  
        self.out_sig_noise = mp.Signal.noise(fs = self.fs,
                                             dur = self.dur,
                                             amp = self.amp/2,
                                             freq_min = self.freq_min,
                                             freq_max = self.freq_max,
                                             unit = "V",
                                             cal = 1.0,
                                             dbfs = self.out_dbfs,
                                             desc = "output_noise")

        self.in_sig_pressure = mp.Signal(fs = self.fs,
                                         unit = "Pa",
                                         dbfs = self.in_dbfs,
                                         cal = self.in_cal,
                                         desc = "input_pressure")
        
        self.in_sig_output = mp.Signal(fs = self.fs,
                                       unit = "V",
                                       dbfs = self.in_dbfs,
                                       cal = 1.0,
                                       desc = "input_output")
        
        #MEASUREMENT        
        self.measurement_noise = mp.Measurement(out_sig = [deepcopy(self.out_sig_noise) for _ in range(4)],
                                                    out_map = [1,2,3,4],
                                                    in_sig = [deepcopy(self.in_sig_pressure),
                                                              deepcopy(self.in_sig_output),
                                                              deepcopy(self.in_sig_output)],
                                                    in_map = [1,3,4],
                                                    dur = self.dur,
                                                    out_device = self.device,
                                                    in_device = self.device,
                                                    device_type = "audio")
        self.measurement_noise.sync_prepare(2,self.added_time)

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
            audio_run_measurement(self.measurement_noise)
            self.measurement_noise.to_dir(self.measurementServerStorageFolder+"measurement_"+str(self.measurementServerCounter))

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
