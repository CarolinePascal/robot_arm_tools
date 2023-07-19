#!/usr/bin/env python3.8

## Definition file of the SoundMeasurementServer class
#
# Defines the attributes and methods used to trigger a sound measurement

import rospy

import measpy as mp
from measpy.audio import audio_run_measurement
import subprocess

from robot_arm_tools import MeasurementServer

## SoundMeasurementServer
#
# Defines the attributes and methods used to trigger a sound measurement
class SoundMeasurementServer(MeasurementServer) :
    
    ## Constructor
    def __init__(self):
        super(SoundMeasurementServer, self).__init__()

        self.M1 = mp.Measurement(out_sig='logsweep',
                                fs=96000,
                                out_sig_freqs=[10,25000],
                                out_map=[1],
                                out_desc=['Sent signal'],
                                out_dbfs=[1.5552],
                                in_map=[1,4,2,3],
                                out_amp=1.0,
                                in_desc=['Pressure','Sent signal','Current','Voltage'],
                                in_cal=[1.0,1.0,0.1,0.1],
                                in_unit=['Pa','V','A','V'],
                                in_dbfs=[1.7108,1.7108,1.7108,1.7108],
                                extrat=[0,0],
                                out_sig_fades=[0,0],
                                dur=10,
                                io_sync=0,
                                in_device=5,
                                out_device=5)

        self.M2 = mp.Measurement(out_sig='noise',
                                fs=96000,
                                out_sig_freqs=[10,25000],
                                out_map=[1],
                                out_desc=['Sent signal'],
                                out_dbfs=[1.5552],
                                in_map=[1,4,2,3],
                                out_amp=1.0,
                                in_desc=['Pressure','Sent signal','Current','Voltage'],
                                in_cal=[1.0,1.0,0.1,0.1],
                                in_unit=['Pa','V','A','V'],
                                in_dbfs=[1.7108,1.7108,1.7108,1.7108],
                                extrat=[0,0],
                                out_sig_fades=[0,0],
                                dur=10,
                                io_sync=0,
                                in_device=5,
                                out_device=5)

    ## Method triggering ALSA drivers recovery
    def recovery(self):

        #sudo chmod -R a+rw /var/run/alsa/
        subprocess.call("pulseaudio -k && /sbin/alsa force-reload", shell=True)
        return(True)

    ## Method triggering a sound measurement
    def measure(self):
        
        #Delay used to avoid sound card Alsa related bugs...
        rospy.sleep(1.0)

        audio_run_measurement(self.M1,progress=False)

        #[Debug] Plot measurement
        #self.M1.plot()
        #plt.show()
        
        self.M1.to_csvwav(self.measurementServerStorageFolder+"sweep_measurement_"+str(self.measurementCounter))

        #Delay used to avoid sound card Alsa related bugs...
        rospy.sleep(2.0)

        audio_run_measurement(self.M2,progress=False)

        #[Debug] Plot measurement
        #self.M2.plot()
        #plt.show()
        
        self.M2.to_csvwav(self.measurementServerStorageFolder+"noise_measurement_"+str(self.measurementCounter))
        
        return(True)

if __name__ == "__main__":
    
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
