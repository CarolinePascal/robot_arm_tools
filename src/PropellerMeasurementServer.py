#!/usr/bin/env python3

## Definition file of the PropellerMeasurementServer class
#
# Defines the attributes and methods used to trigger a propeller measurement
#
# Do not forget to set the following environment variables : 
# export ROS_MASTER_URI=http://this_computer_IP_address:11311 
# export ROS_IP=this_computer_IP_address 

import rospy
from std_srvs.srv import Empty,EmptyResponse

import os
import threading

import measpy as mp
from measpy.audio import audio_run_measurement

## PropellerMeasurementServer
#
# Defines the attributes and methods used to trigger a propeller measurement
class PropellerMeasurementServer :
    
    ## Constructor
    def __init__(self):

        ## Propeller sound measurement
        self.M1 = mp.Measurement(in_map=[1],
                    in_desc=['In1'],
                    in_cal=[1.0],
                    in_unit=['Pa'],
                    in_dbfs=[1.0/0.593],
                    extrat=[0.0,0.0],
                    out_sig_fades=[0.0,0.0],
                    dur=5+1)

        ## Storage folder name
        self.storageFolderName = rospy.get_param("storageFolderName")
        try:
            os.mkdir(self.storageFolderName)
            rospy.loginfo("The folder " + self.storageFolderName + "was created")
        except OSError:
            rospy.logwarn("The folder " + self.storageFolderName + "already exists : its contents will be ereased !")
            pass 

        ## ROS Service Server
        self.microServer = rospy.Service("/propeller_measurement_server",Empty,self.measure)

        ## Measurement counter 
        self.measurementCounter = 0

        # ROS Service Client
        rospy.wait_for_service("propeller_server")

    ## Method triggering a propeller measurement
    #  @param req An empty ROS service request
    def measure(self, req):
        
        self.measurementCounter += 1

        #Delay used to avoid sound card Alsa related bugs...
        rospy.sleep(2.0)

        #Start the measurement on an independant, non-blocking thread
        thread = threading.Thread(target=audio_run_measurement, args=(self.M1,))
        thread.start()
        rospy.sleep(0.5)

        #Send the starting command to the propeller
        rospy.ServiceProxy("propeller_server", Empty)()
        rospy.sleep(0.5)

        #Wait for the measurement to be over
        thread.join()

        #[Debug] Plot measurement
        #self.M1.plot()
        #plt.show()

        #Save measurement results
        self.M1.to_csvwav(self.storageFolderName+"noise_measurement"+str(self.measurementCounter))

        return EmptyResponse()

def main():
    #Launch ROS node
    rospy.init_node('propeller_measurement_server')

    #Launch ROS service
    PropellerMeasurementServer()

    while not rospy.is_shutdown():
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("Shutting down ROS propeller measurement server")
            break

if __name__ == "__main__":
    main()