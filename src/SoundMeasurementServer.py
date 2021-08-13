#!/usr/bin/env python3

## Definition file of the SoundMeasurementServer class
#
# Defines the attributes and methods used to trigger a sound measurement

import rospy
from std_srvs.srv import Empty,EmptyResponse

import os

import measpy as mp
from measpy.audio import audio_run_measurement

## SoundMeasurementServer
#
# Defines the attributes and methods used to trigger a sound measurement
class SoundMeasurementServer :
    
    ## Constructor
    def __init__(self):

        ## First sound measurement
        self.M1 = mp.Measurement(out_sig='noise',
                    out_map=[4],
                    out_desc=['Out4'],
                    out_dbfs=[1.0/1.53],
                    in_map=[1],
                    in_desc=['In1'],
                    in_cal=[1.0],
                    in_unit=['Pa'],
                    in_dbfs=[1.0/0.593],
                    extrat=[0.0,0.0],
                    out_sig_fades=[0.0,0.0],
                    dur=5)

        ## Second sound measurement
        self.M2 = mp.Measurement(out_sig='logsweep',
                    out_map=[4],
                    out_desc=['Out4'],
                    out_dbfs=[1.0/1.53],
                    in_map=[1],
                    in_desc=['In1'],
                    in_cal=[1.0],
                    in_unit=['Pa'],
                    in_dbfs=[1.0/0.593],
                    extrat=[0.0,0.0],
                    out_sig_fades=[0.0,0.0],
                    dur=5)

        ## Storage folder name
        self.storageFolderName = rospy.get_param("storageFolderName")
        try:
            os.mkdir(self.storageFolderName)
            rospy.loginfo("Creating " + self.storageFolderName + " ...")
        except OSError:
            rospy.logwarn(self.storageFolderName + "already exists : its contents will be overwritten !")
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
        rospy.sleep(2.0)

        #Run measurement #1
        audio_run_measurement(self.M1)

        #[Debug] Plot measurement
        #self.M1.plot()
        #plt.show()

        #Save measurement #1 results
        self.M1.to_csvwav(self.storageFolderName+"noise_measurement"+str(self.measurementCounter))

        #Delay used to avoid sound card Alsa related bugs...
        rospy.sleep(2.0)

        #Run measurement #2
        audio_run_measurement(self.M2)

        #[Debug] Plot measurement
        #self.M2.plot()
        #plt.show()

        #Save measurement #2 results
        self.M2.to_csvwav(self.storageFolderName+"sweep_measurement"+str(self.measurementCounter))

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