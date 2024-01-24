#!/usr/bin/python3

#Acoustics package
import measpy as ms
from measpy._tools import wrap

#Utility packages
import numpy as np
import glob
import os
import sys
import csv
from copy import copy

#Mesh package
import trimesh as trimesh
import meshio as meshio

#Plot package
import matplotlib.pyplot as plt

from DataProcessingTools import plot_3d_data, save_fig, set_title, fmin, fmax, octBand, figsize, octBandFrequencies

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname((os.path.abspath(__file__))))) + "/scripts")
from MeshTools import plotMesh, plotPointCloudFromPath

def sphereFit(points):
    A = np.zeros((len(points),4))
    A[:,0] = points[:,0]*2
    A[:,1] = points[:,1]*2
    A[:,2] = points[:,2]*2
    A[:,3] = 1

    #   Assemble the f matrix
    f = np.zeros((len(points),1))
    f[:,0] = (points[:,0]**2) + (points[:,1]**2) + (points[:,2]**2)
    output, residules, rank, singval = np.linalg.lstsq(A,f)

    #   solve for the radius
    radius = np.sqrt(output[0]**2 + output[1]**2 + output[2]**2 + output[3])
    center = output[:3].T

    return(center, radius)

if __name__ == "__main__":

	#Get processing method 
	processingMethod = "welch"
	try:
		processingMethod = sys.argv[1].lower()
		if(processingMethod not in ["welch","farina"]):
			raise ValueError("Invalid processing method")
	except IndexError:
		print("Invalid processing method, defaulting to " + processingMethod + " method")

	#Get interest output signal type
	outputSignalType = "sweep"
	try:
		outputSignalType = sys.argv[2].lower()
	except IndexError:
		print("Invalid output signal type, defaulting to " + str(outputSignalType) + " output signal")

	if(not os.path.isdir(processingMethod + "_" + outputSignalType)):
		os.mkdir(processingMethod + "_" + outputSignalType)

	#Get transfer function input and output signals names
	inputSignal = "Out1" #Voltage
	outputSignal = "In1"   #Pressure
	try:
		inputSignal = sys.argv[3]
		outputSignal = sys.argv[4]
	except IndexError:
		print("Invalid input/output signals, defaulting to input : " + inputSignal + " and output : " + outputSignal)

	#Get interest frequencies
	Frequencies = octBandFrequencies
	try:
		Frequencies = [int(item) for item in sys.argv[5].split(",")]
	except (IndexError, ValueError):
		print("Invalid frequency, defaulting to f = " + str(Frequencies) + " Hz")

	print("Processing input " + inputSignal + " and output " + outputSignal + " with " + processingMethod + " and " + outputSignalType + " output signal at " + str(Frequencies)  + " Hz")

	#Get point cloud path
	pointCloudPath = None
	try:
		pointCloudPath = sys.argv[6]
	except IndexError:
		pass
	if(not os.path.isfile(pointCloudPath)):
		pointCloudPath = None

	#Get mesh path
	meshPath = ""
	meshPathDefault = "initMesh.mesh"
	try:
		meshPath = sys.argv[7]
		if(not os.path.isfile(meshPath)):
			meshPath = meshPathDefault
			if(not os.path.isfile(meshPath)):
				print("Invalid mesh path, defaulting to no mesh data")
				meshPath = None
			else:
				print("Invalid mesh path, defaulting to " + meshPathDefault)
	except IndexError:
		pass

	#Get mesh element type
	elementType = "P0"
	try:
		elementType = sys.argv[8]	
	except IndexError:
		if(meshPath is not None):
			print("Invalid element type, defaulting to " + elementType)

	Files = sorted(glob.glob(outputSignalType + "*.wav"), key=lambda file:int(os.path.basename(file).split(".")[0].split("_")[-1]))
	
	Data = np.empty((len(Frequencies),len(Files)),dtype=complex)

	for i,file in enumerate(Files):

		print("Data processing file : " + file)
		M = ms.Measurement.from_csvwav(file.split(".")[0])
		
		#Check processing method compatibility
		if(processingMethod == "farina" and M.out_sig != "logsweep"):
			raise ValueError("Farina method cannot be used with non log sweep signals")

		P = M.data[outputSignal]
		V = M.data[inputSignal]
		
		TFE = None
		if(processingMethod == "farina"):
			TFE = P.tfe_farina([fmin,fmax])
		else:
			TFE = P.tfe_welch(V) #Also possible for dB values : (P*V.rms).tfe_welch(V)
	
		for j,f in enumerate(Frequencies):
			Data[j,i] = TFE.nth_oct_smooth_to_weight_complex(octBand,fmin=f,fmax=f).acomplex[0]


	X = []
	Y = []
	Z = []

	with open("States.csv", newline='') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		for row in reader:
			X.append(float(row[0]))
			Y.append(float(row[1]))
			Z.append(float(row[2]))

	X = np.array(X)
	Y = np.array(Y)
	Z = np.array(Z)

	Points = np.array([X,Y,Z]).T

	#To adapt depending on the mesh type ! 
	#centroid = np.array([0.4419291797546691,-0.012440529880238332,0.5316684442730065])
	#centroid = np.mean(Points,axis=0)
	centroid, _ = sphereFit(Points)

	if(meshPath is None):

		for f,data in zip(Frequencies,Data):

			print("Processing frequency " + str(int(f)) + " Hz")

			figAmp,axAmp = plt.subplots(1,figsize=figsize,subplot_kw=dict(projection='3d'))
			figPhase,axPhase = plt.subplots(1,figsize=figsize,subplot_kw=dict(projection='3d'))
			plot_3d_data(np.abs(data), Points.T, axAmp, label = r"$|$H$|$ (Pa/V)")
			plot_3d_data(wrap(np.angle(data)), Points.T, axPhase, label = "Phase (rad)")

			if(pointCloudPath is not None):
				plotPointCloudFromPath(pointCloudPath, axAmp, s=5)
				plotPointCloudFromPath(pointCloudPath, axPhase, s=5)

			#set_title(axAmp,"Pressure/Input signal TFE amplitude at " + str(int(f)) + " Hz\n1/" + str(octBand) + " octave smoothing")
			#set_title(axPhase,"Pressure/Input signal TFE phase at " + str(int(f)) + " Hz\n1/" + str(octBand) + " octave smoothing")
			#axAmp.set_title("Measured amplitude at " + str(int(f)) + " Hz")
			#axPhase.set_title("Measured phase at " + str(int(f)) + " Hz")
			save_fig(figAmp, processingMethod + "_" + outputSignalType + "/amplitude_" + str(int(f)) + ".pdf")
			save_fig(figPhase, processingMethod + "_" + outputSignalType + "/phase_" + str(int(f)) + ".pdf")
			plt.close("all")

			np.savetxt(processingMethod + "_" + outputSignalType + "/data_" + str(int(f)) + ".csv",np.array([np.real(data),np.imag(data)]).T,delimiter=",")
			
		if(not os.path.exists("robotMesh.mesh")):
			meshio.write_points_cells("robotMesh.mesh", Points, [("line",np.vstack((np.arange(len(Points)),np.roll(np.arange(len(Points)),-1))).T)])
  
	else:

		try:
			import shutil
			shutil.copyfile(meshPath,"initMesh.mesh")
		except shutil.SameFileError:
			pass
			
		mesh = meshio.read("initMesh.mesh")

		Vertices, Faces = mesh.points, mesh.get_cells_type("triangle")
		detailedMesh = trimesh.Trimesh(Vertices, Faces)

		resolution = np.mean(detailedMesh.edges_unique_length)
		Centroids = detailedMesh.triangles_center

		MeasurementsVertices = Vertices + centroid
		MeasurementsCentroids = Centroids + centroid
  
		if(elementType == "P0"):
			MeasurementsPoints = MeasurementsCentroids
		elif(elementType == "P1"):
			MeasurementsPoints = MeasurementsVertices
		else:
			raise ValueError("Invalid element type")

		missing = []
		OrderedData = np.empty((len(Frequencies),len(MeasurementsPoints)),dtype=complex)

		for i,MeasurementsPoint in enumerate(MeasurementsPoints):

				distances = np.linalg.norm(Points - MeasurementsPoint,axis=1)
				indexMin = np.argmin(distances)

				#If the closest point is too far away, we consider that the Measurements point is missing
				if(distances[indexMin] > resolution/2):
						print("Missing measurement detected at point " + str(MeasurementsPoint) + " (" + str(distances[indexMin]) + " m)")
						missing.append(i)
				else:
						OrderedData[:,i] = Data[:,indexMin]

		for i in missing:
				closest = np.argsort(np.linalg.norm(Points - MeasurementsPoints[i], axis=1))[1:4]
				OrderedData[:,i] = np.average(np.abs(Data[:,closest]),axis=1)*np.exp(1j*np.average(np.angle(Data[:,closest]),axis=1))
  
		for f,data in zip(Frequencies,OrderedData):

			print("Processing frequency " + str(int(f)) + " Hz")

			figAmp,axAmp = plt.subplots(1,figsize=figsize,subplot_kw=dict(projection='3d'))
			figPhase,axPhase = plt.subplots(1,figsize=figsize,subplot_kw=dict(projection='3d'))
			plot_3d_data(np.abs(data), MeasurementsPoints.T, axAmp, label = r"$|$H$|$ (Pa/V)")
			plot_3d_data(wrap(np.angle(data)), MeasurementsPoints.T, axPhase, label = "Phase (rad)")

			if(pointCloudPath is not None):
				plotPointCloudFromPath(pointCloudPath, axAmp, s=5)
				plotPointCloudFromPath(pointCloudPath, axPhase, s=5)

			if(meshPath is not None):
				plotMesh(MeasurementsVertices, Faces, ax = axAmp, linewidth=2)
				plotMesh(MeasurementsVertices, Faces, ax = axPhase, linewidth=2)

			#set_title(axAmp,"Pressure/Input signal TFE amplitude at " + str(int(f)) + " Hz\n1/" + str(octBand) + " octave smoothing")
			#set_title(axPhase,"Pressure/Input signal TFE phase at " + str(int(f)) + " Hz\n1/" + str(octBand) + " octave smoothing")
			#axAmp.set_title("Pressure/Input signal TFE amplitude at " + str(int(f)) + " Hz")
			#axPhase.set_title("Pressure/Input signal TFE phase at " + str(int(f)) + " Hz")
			save_fig(figAmp, processingMethod + "_" + outputSignalType + "/amplitude_" + str(int(f)) + ".pdf")
			save_fig(figPhase, processingMethod + "_" + outputSignalType + "/phase_" + str(int(f)) + ".pdf")
			plt.close("all")
  
			np.savetxt(processingMethod + "_" + outputSignalType + "/data_" + str(int(f)) + ".csv",np.array([np.real(data),np.imag(data)]).T,delimiter=",")

		if(not os.path.exists("robotMesh.mesh")):
			meshio.write_points_cells("robotMesh.mesh", MeasurementsVertices, mesh.cells)
  
	#plt.show()