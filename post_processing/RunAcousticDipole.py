import subprocess
from SphericMesh import createMesh
import numpy as np
import matplotlib.pyplot as plt

Rmin = 0.2
Ntheta = 10
Nphi = 10

DeltaR = np.logspace(-3,0,100)

Error = []

for delta in DeltaR:
    Rmax = np.round(Rmin + delta,5)

    print("Rmin : " + str(Rmin) + " m")
    print("Rmax : " + str(Rmax) + " m")

    createMesh(Ntheta,Nphi,Rmin,Rmax)

    bashCommand = "FreeFem++ AcousticDipole.edp -angularVerticesNumber " +str(Ntheta) + " -Rmin " + str(Rmin) + " -Rmax " + str(Rmax)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    msg = output.decode()
    error = float(msg[-msg[::-1].find('\n\n'):-1])
    Error.append(error)
    
figLog, axLog = plt.subplots()

axLog.plot(DeltaR,Error)
axLog.set_xlabel("Radius delta (m)")
axLog.set_ylabel("Average error (Pa)")
axLog.set_title("Average error depending on the radius delta")    
axLog.legend()
plt.show()
