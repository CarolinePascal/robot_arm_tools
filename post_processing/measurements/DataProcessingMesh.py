import measpy as ms

import glob
import os
import sys

import csv

import meshio as meshio

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.cm as cm
import matplotlib.colors as colors

cmap = plt.get_cmap("tab10")

plt.rc('font', **{'size': 12, 'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)

import open3d as o3d

Data = []   
#fmin = 150  #Anechoic room cutting frquency
#fmax =  10000   #PU probe upper limit
fmin = 20
fmax = 20000
input = "sweep"

f = 500
try:
        f = int(sys.argv[1])
        print("Setting f = " + str(f) + " Hz")
except:
        print("Defaulting to f = " + str(f) + " Hz")

show = False
try:
        show = sys.argv[2]
except:
        pass

octBand = 12

Files = sorted(glob.glob(input + "*.wav"), key=lambda file:int(os.path.basename(file).split(".")[0].split("_")[-1]))

for i,file in enumerate(Files):

        print("Data processing file : " + file)
        M = ms.Measurement.from_csvwav(file.split(".")[0])

        p = M.data["In1"]
        v = M.data["In4"]

        Data.append(p.tfe_welch(v).nth_oct_smooth_to_weight_complex(octBand,fmin=f,fmax=f).acomplex[0])

Data = np.array(Data)

X = []
Y = []
Z = []

with open("States.csv", newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in reader:
        X.append(float(row[0]))
        Y.append(float(row[1]))
        Z.append(float(row[2]))

X = np.array(X)
Y = np.array(Y)
Z = np.array(Z)

points = np.array([X,Y,Z]).T

# DEBUG
#Data = np.random.random(len(X))*10 + 1j*np.random.random(len(X))*10

pointCloud = o3d.io.read_point_cloud("/media/caroline/Transcend/scans/SphericScanJBL2_none_true/FinalPointCloud.pcd") 
pointCloud = pointCloud.voxel_down_sample(voxel_size=0.005)
pointCloudPoints = np.asarray(pointCloud.points)
pointCloudColors = np.asarray(pointCloud.colors)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(*pointCloudPoints.T,c=pointCloudColors,s=2)

sc = ax.scatter(X,Y,Z,c=np.abs(Data),cmap="jet")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_zlabel("z (m)")
ax.set_title("Acoustic pressure field amplitude (Pa) at " + str(f) + " Hz")
plt.colorbar(sc,pad=0.1)

if(show):
        plt.show()

fig.tight_layout()
fig.savefig("./amplitude_data_" + str(f) + "_points.pdf",bbox_inches='tight')

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(*pointCloudPoints.T,c=pointCloudColors,s=2)

sc = ax.scatter(X,Y,Z,c=20*np.log10(np.abs(Data)/ms.PREF.v),cmap="jet")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_zlabel("z (m)")
ax.set_title("Acoustic pressure field amplitude (dB) at " + str(f) + " Hz")
plt.colorbar(sc,pad=0.1)

if(show):
        plt.show()

fig.tight_layout()
fig.savefig("./spl_data_" + str(f) + "_points.pdf",bbox_inches='tight')

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(*pointCloudPoints.T,c=pointCloudColors,s=2)

sc = ax.scatter(X,Y,Z,c=np.angle(Data),cmap="jet")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_zlabel("z (m)")
ax.set_title("Acoustic pressure field phase (rad) at " + str(f) + " Hz")
plt.colorbar(sc,pad=0.1)

if(show):
        plt.show()

fig.tight_layout()
fig.savefig("./phase_data_" + str(f) + "_points.pdf",bbox_inches='tight')

#Not working...
#sys.path.append(os.getcwd())
#from scatter import scatter
#plotValues = 20*np.log10(np.abs(Data)/ms.PREF.v)
#norm = colors.Normalize(vmin=min(plotValues), vmax=max(plotValues), clip=True)
#mapper = cm.ScalarMappable(norm=norm, cmap=cm.jet)
#scatter(X,Y,Z,C=mapper.to_rgba(plotValues))

### Load mesh
centerPosition = np.array([0.4419291797546691,-0.012440529880238332,0.5316684442730065])

resolution = float(os.getcwd().split('_')[-1])
mesh = meshio.read("./mesh.mesh")

vertices, faces = mesh.points, mesh.cells_dict["triangle"]
centroids = np.average(vertices[faces],axis=1)

measurementVertices = vertices + centerPosition
measurementPoints = centroids + centerPosition

missing = []
orderedData = []
orderedPoints = []

for i,measurementPoint in enumerate(measurementPoints):

        if(np.min(np.linalg.norm(points-measurementPoint,axis=1)) > resolution/2):
                missing.append(i)
                orderedData.append(None)
                orderedPoints.append(measurementPoint)
        else:
                orderedData.append(Data[np.argmin(np.linalg.norm(points-measurementPoint,axis=1))])
                orderedPoints.append(points[np.argmin(np.linalg.norm(points-measurementPoint,axis=1))])

for i in missing:
        #print(i)
        closest = np.argsort(np.linalg.norm(points - measurementPoints[i], axis=1))[1:4]
        #print(Data[closest])
        orderedData[i] = np.average(np.abs(Data[closest]))*np.exp(1j*np.average(np.angle(Data[closest])))

orderedData = np.array(orderedData,dtype=complex)
np.savetxt("data_" + str(f) + ".csv",np.array([np.real(orderedData),np.imag(orderedData)]).T,delimiter=",")

orderedPoints = np.array(orderedPoints)
if(not os.path.exists("./robotMesh.mesh")):
        meshio.write_points_cells("./robotMesh.mesh", orderedPoints-centerPosition, mesh.cells)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

plotValues = 20*np.log10(np.abs(orderedData)/ms.PREF.v)
norm = colors.Normalize(vmin=min(plotValues), vmax=max(plotValues), clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.jet)

ts = ax.plot_trisurf(measurementVertices[:,0],measurementVertices[:,1],measurementVertices[:,2],triangles=faces, antialiased=True, linewidth=0.1, edgecolor='black')
ts.set_facecolors(mapper.to_rgba(plotValues))

ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_zlabel("z (m)")
ax.set_title("Acoustic pressure field amplitude (dB) at " + str(f) + " Hz")

plt.colorbar(mapper,pad=0.1)

if(show):
        plt.show()

fig.tight_layout()
fig.savefig("./data_" + str(f) + "_faces.pdf",bbox_inches='tight')
