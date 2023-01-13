from warnings import simplefilter
import measpy as ms
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from cmath import *

import csv
import os

from numpy.lib.function_base import append

N = 91
fmin = 20
fmax = 20000
f = 1000

Spheres = ["measurements/17062021_S-","measurements/17062021_S0","measurements/17062021_S+"]

Amp = []
Re = []
Im = []

for sphere in Spheres:

    for i in range(N):

        print("Measurement "+str(i+1)+"/"+str(N)+" of sphere "+sphere)

        M1 = ms.Measurement.from_csvwav(sphere+"/sweep_measurement_0_"+str(i+1))

        S1 = M1.data["In1"]
        #S1dB = S1.dB_SPL()
        #S1dB.plot()
        #plt.show()
        FFT1 = S1.fft()
        #FFT1.plot()
        #plt.show()

        smoothFFT1 = FFT1.nth_oct_smooth_complex(3)
        #smoothFFT1.plot()
        #plt.show()

        #print(len(smoothFFT1.values))
        F = np.array(smoothFFT1.freqs)
        Imin = np.argwhere(np.round(F,2)==fmin)[0][0]
        Imax = np.argwhere(np.round(F)==fmax)[0][0]
        I = np.argwhere(np.round(F,1)==f)[0][0]

        #print(smoothFFT1.values[I])
        #print(20*np.log10(np.abs(smoothFFT1.values[I]/20e-6)))

        Amp.append(20*np.log10(np.abs(smoothFFT1.values[I]/20e-6)))
        Re.append(smoothFFT1.values[I].real)
        Im.append(smoothFFT1.values[I].imag)

for i,sphere in enumerate(Spheres):

    if(os.path.isfile(sphere+"/Pressure"+"_"+str(f)+"_complex.csv")):
        print("Pressure"+"_"+str(f)+"_complex.csv alredy exists - Overwriting !")
        os.remove(sphere+"/Pressure"+"_"+str(f)+"_complex.csv")
    else:
        print("Creating Pressure"+"_"+str(f)+"_complex.csv...")

for j in range(N):

    buffer = np.array([Amp[j],Amp[N+j],Amp[2*N+j]])
    avg = np.mean(buffer)
    std = np.std(buffer)

    print(buffer)
    print(avg)
    print(std)

    for i,sphere in enumerate(Spheres):

        with open(sphere+"/Pressure"+"_"+str(f)+"_complex.csv", 'a') as csvfile:

            writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)

            if(np.abs(Amp[i*N+j] - avg) > std and std > 1):
                plusValue = complex(Re[i*N+j+1],Im[i*N+j+1])
                minusValue = complex(Re[i*N+j-1],Im[i*N+j-1])

                avgValue = ((np.abs(plusValue)+np.abs(minusValue))/2)*np.exp(1j*(np.angle(plusValue)+np.angle(minusValue))/2)

                print(np.real(avgValue))
                print(np.imag(avgValue))

                writer.writerow([str(np.real(avgValue)),str(np.imag(avgValue))])

            else:
                writer.writerow([str(Re[i*N+j]),str(Im[i*N+j])])

X = []
Y = []
Z = []

for sphere in Spheres:

    with open(sphere+"/Positions.csv", newline='') as csvfile:

        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        counter = 1
        for row in reader:
            if(counter > N):
                break
            counter += 1
            X.append(float(row[0]))
            Y.append(float(row[1]))
            Z.append(float(row[2]))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(X,Y,Z,c=Amp,cmap="jet")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Acoustic pressure field amplitude (dB)\nValue for 1 kHz")
plt.colorbar(sc)
plt.show()
"""

X = []
Y = []
Z = []
Amp = []

for sphere in Spheres:

    with open(sphere+"/Positions.csv", newline='') as csvfile:

        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        counter = 1
        for row in reader:
            if(counter > N):
                break
            counter += 1
            X.append(float(row[0]))
            Y.append(float(row[1]))
            Z.append(float(row[2]))

    with open(sphere+"/Pressure"+"_"+str(f)+"_complex.csv", newline='') as csvfile:

        reader = csv.reader(csvfile, delimiter=',', quotechar='|')

        for row in reader:
            P = complex(float(row[0]),float(row[1]))
            Amp.append(20*np.log10(np.abs(P/20e-6)))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(X,Y,Z,c=Amp,cmap="jet")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Acoustic pressure field amplitude (dB)\nValue for 1 kHz")
plt.colorbar(sc)
plt.show()

imin = np.argmin(Amp)
print(Amp[imin-N],Amp[imin],Amp[imin+N])
"""