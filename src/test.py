#!/usr/bin/env python3.8

## Definition file of the MicrophoneCalibrationServer class
#
# Defines the attributes and methods used to trigger a microphone calibration measurement

import matplotlib.pyplot as plt

import measpy as mp
from measpy.audio import audio_run_measurement

import numpy as np
import sys

refAmp = float(sys.argv[1])
refFreq = float(sys.argv[2])

flag = False
initInCal = 1.0

M0 = mp.Measurement(out_sig=None,
                    fs=48000,
                    in_map=[1],
                    in_desc=['Pressure'],
                    in_cal=[0.316],
                    in_unit=['Pa'],
                    in_dbfs=[1.0/0.593],
                    dur=2,
                    in_device=4)

while(not flag):

    #audio_run_measurement(M0)
    M0.data['In1'].values = 3*np.sin(2*np.pi*np.linspace(0,M0.dur,M0.fs*M0.dur)*1000)
    #M0.plot()
    #plt.show()

    print(M0.data['In1'].rms()*np.sqrt(2))

    fft0 = M0.data['In1'].rfft(norm="forward")
    fft0.values *= 2
    #fft0.plot()
    #plt.show()

    dBfft0 = np.abs(fft0.values)
    argMaxAmp = np.argmax(dBfft0)
    maxAmp = dBfft0[argMaxAmp]
    maxFreq = fft0.freqs[argMaxAmp]

    print("Maximum amplitude " + str(np.round(maxAmp,3)) + " dB recorded at " + str(np.round(maxFreq,3)) + " Hz")

    if(np.round(maxFreq) == refFreq):
        print("Frequency check : OK")
    else:
        print("Prout")

    if(np.round(maxAmp) != refAmp):
        newInCal = 10**((refAmp - maxAmp)/20)
        print("Trying new calibration : " + str(np.round(M0.in_cal[0]/newInCal,3)) + " V/Pa")
        M0 = mp.Measurement(out_sig=None,
                    fs=48000,
                    in_map=[1],
                    in_desc=['Pressure'],
                    in_cal=[M0.in_cal[0]/newInCal],
                    in_unit=['Pa'],
                    in_dbfs=[1.0/0.593],
                    dur=10,
                    in_device=4)
    else:
        print("Calibration check : OK")
        print("Final calibration to be set : " + str(np.round(M0.in_cal[0],3)) + " V/Pa")
        flag = True

