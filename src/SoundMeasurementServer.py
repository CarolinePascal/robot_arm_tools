#!/usr/bin/env python3

### Sound measurements ###

import rospy
from std_srvs.srv import Empty,EmptyResponse

import os

import measpy as mp
from measpy.audio import audio_run_measurement

class SoundMeasurementServer :
    
    def __init__(self):
        ### Measurements

        self.M1 = mp.Measurement(out_sig='logsweep',
                    out_amp=0.4,
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

        self.M2 = mp.Measurement(out_sig='logsweep',
                    out_amp=1,
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

        ### Data 

        #Storage folder
        self.storageFolder = "/tmp/SoundMeasurements/"
        try:
            os.mkdir(self.storageFolder)
            rospy.loginfo("The folder " + self.storageFolder + "was created")
        except OSError:
            rospy.logwarn("The folder " + self.storageFolder + "already exists : its contents will be ereased !")
            pass 

        ### ROS Service Server

        self.microServer = rospy.Service("/sound_measurement_server",Empty,self.measure)
        self.measureCounter = 0

    def measure(self, req):
        self.measureCounter += 1

        rospy.sleep(2.0)
        audio_run_measurement(self.M1)
        #self.M1.plot()
        #plt.show()
        self.M1.to_csvwav(self.storageFolder+"sweep_measurement_0_"+str(self.measureCounter))

        rospy.sleep(2.0)
        audio_run_measurement(self.M2)
        #self.M2.plot()
        #plt.show()
        self.M2.to_csvwav(self.storageFolder+"sweep_measurement_1_"+str(self.measureCounter))

        return EmptyResponse()

def main():
    rospy.init_node('sound_measurement_server')

    SoundMeasurementServer()

    while not rospy.is_shutdown():
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("Shutting down ROS sound measurement server")
            break

if __name__ == "__main__":
    main()