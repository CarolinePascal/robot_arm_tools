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
from copy import deepcopy

#Mesh package
import trimesh as trimesh
import meshio as meshio

#Point cloud package
import open3d as o3d

#Plot package
import matplotlib.pyplot as plt
import plotly.graph_objects as go

#Multiprocessing package
from multiprocessing import Pool

from robot_arm_acoustic_post_processing.measurements.PlotTools import plot_3d_data, save_fig, set_title, fmin, fmax, octBand, figsize, octBandFrequencies

from robot_arm_acoustic.MeshTools import plotMesh, plotPointCloudFromPath

INTERACTIVE = False

def sphereFit(points):
    A = np.zeros((len(points),4))
    A[:,0] = points[:,0]*2
    A[:,1] = points[:,1]*2
    A[:,2] = points[:,2]*2
    A[:,3] = 1

    #   Assemble the f matrix
    f = np.zeros((len(points),1))
    f[:,0] = (points[:,0]**2) + (points[:,1]**2) + (points[:,2]**2)
    output, residules, rank, singval = np.linalg.lstsq(A,f,rcond=None)

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
	if(not pointCloudPath is None and not os.path.isfile(pointCloudPath)):
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

	#Wether to adapt data to the mesh (i.e. when data is not provided at the centroids locations)
	adaptToMesh = False
	try:
		adaptToMesh = bool(int(sys.argv[9]))
		if(adaptToMesh and elementType != "P0"):
			raise ValueError("Adapt to mesh is only possible with P0 elements")
	except IndexError:
		pass

	if(adaptToMesh):
		print("[WARNING] Data will be adapted to the mesh")

	#Retrieve measurements data and locations

	folderName = processingMethod + "_" + outputSignalType + "_" + inputSignal + "_" + outputSignal + "_" + elementType
	os.makedirs(folderName,exist_ok=True)

	Files = sorted(glob.glob(outputSignalType + "*.wav"), key=lambda file:int(os.path.basename(file).split(".")[0].split("_")[-1]))
	
	Data = np.empty((len(Frequencies),len(Files)),dtype=complex)

	def processing_data(file):
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
		
		output = np.empty(len(Frequencies),dtype=complex)
		for j,f in enumerate(Frequencies):
			output[j] = TFE.nth_oct_smooth_to_weight_complex(octBand,fmin=f,fmax=f).acomplex[0]
			#output[j] = 100.0

		return(output)

	with Pool(os.cpu_count()-1) as pool:
		Data = np.array(pool.map(processing_data,Files)).T

	X = []
	Y = []
	Z = []

	with open("States.csv", 'r') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		for row in reader:
			X.append(float(row[0]))
			Y.append(float(row[1]))
			Z.append(float(row[2]))

	X = np.array(X)
	Y = np.array(Y)
	Z = np.array(Z)

	Points = np.array([X,Y,Z]).T

	#Measurement mesh
	if(not meshPath is None):

		### TO ADPAT DEPENDING ON MESH TYPE ###
		#centroid = np.array([0.4419291797546691,-0.012440529880238332,0.5316684442730065])
		#centroid = np.mean(Points,axis=0)
		centroid, radius = sphereFit(Points)
		print("Estimated sphere centroid : " + str(centroid) + " m")
		print("Estimated sphere radius : " + str(radius) + " m")
		########################################
		
		mesh = meshio.read(meshPath)
		Vertices, Faces = mesh.points, mesh.get_cells_type("triangle")

		### TO ADPAT DEPENDING ON MESH TYPE ###
		#Shift vertices and centroids into the real world frame (measurements frame)
		MeasurementsVertices = Vertices + centroid
		########################################

		#Compute measurements resolution
		pointCloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Points))
		resolution = np.array(pointCloud.compute_nearest_neighbor_distance()).max()

		#Compute mesh resolution
		detailedMesh = trimesh.Trimesh(MeasurementsVertices, Faces)
		meshResolution = detailedMesh.edges_unique_length.max()

		#Set the measurements points
		if(elementType == "P0"):
			MeasurementsPoints = np.mean(MeasurementsVertices[Faces],axis=1)
		elif(elementType == "P1"):
			MeasurementsPoints = MeasurementsVertices
		else:
			raise ValueError("Invalid element type")

		OrderedData = np.empty((len(Frequencies),len(MeasurementsPoints)),dtype=complex)

		if(adaptToMesh):

			#Get mesh closest face for each measurement point
			_,_,faceIndex = trimesh.proximity.closest_point(detailedMesh,Points) 
			DataBuffer = []
			sortedFaceIndex = np.sort(faceIndex)
			sortedData = Data[:,np.argsort(faceIndex)]

			#For each face, compte the average value of the closest measurements
			previousIndex = sortedFaceIndex[0]
			for i,(index,data) in enumerate(zip(sortedFaceIndex,sortedData.T)):
				if(index != previousIndex):
					OrderedData[:,previousIndex] = np.average(np.abs(DataBuffer),axis=0)*np.exp(1j*np.average(np.angle(DataBuffer),axis=0))
					DataBuffer = []
					previousIndex = index
				DataBuffer.append(data)

			#Compute missing points (i.e. faces with no closest measurement point)
			missing = [i for i in range(len(MeasurementsPoints)) if i not in faceIndex]
			for i in missing:
				print("Missing measurement detected at point " + str(i) + " : " + str(MeasurementsPoints[i]))
				
		else:

			missing = []
			
			#Reorder data according to the mesh and find missing measurements
			for i,MeasurementsPoint in enumerate(MeasurementsPoints):

				#Find the closest measurement point to the current mesh point
				distances = np.linalg.norm(Points - MeasurementsPoint,axis=1)
				indexMin = np.argmin(distances)

				#If the closest measurement point is too far away, we consider that the Measurements point is missing
				#Else, we assign the corresponding data value to the mesh point
				print(distances[indexMin])
				if(distances[indexMin] > resolution/2):
					print("Missing measurement detected at point " + str(i) + " : " + str(MeasurementsPoint) + " (" + str(distances[indexMin]) + " m)")
					missing.append(i)
				else:
					OrderedData[:,i] = Data[:,indexMin]

		#Fill missing mesh points
		filledPoints = deepcopy(Points)

		#While there are missing points...
		while(len(missing) > 0):

			#Compute the distances to the three closest filled points
			closestPointsIndex = np.empty((len(missing),3),dtype=int)
			criteria = np.zeros(len(missing))

			for i,index in enumerate(missing):
				distances = np.linalg.norm(filledPoints - MeasurementsPoints[index], axis=1)
				closestPointsIndex[i] = np.argsort(distances)[:3]

				closestPointsDistance = distances[closestPointsIndex[i]]
				criteria[i] = len(np.where(closestPointsDistance < resolution)[0])
			
			#Fill the best candidate points with the 3 closest points
			bestIndex = np.where(criteria == np.max(criteria))[0]
			for i in bestIndex:
				OrderedData[:,missing[i]] = np.average(np.abs(Data[:,closestPointsIndex[i]]),axis=1)*np.exp(1j*np.average(np.angle(Data[:,closestPointsIndex[i]]),axis=1))

				#Update list of missing points
				filledPoints = np.vstack((filledPoints,MeasurementsPoints[missing[i]]))
				Data = np.vstack((Data.T,OrderedData[:,missing[i]])).T

			missing = [missing[i] for i in range(len(missing)) if i not in bestIndex]

		#Save mesh, but offsetted to the measurements centroid
		meshio.write_points_cells(folderName + "/robotMesh.mesh", MeasurementsVertices, mesh.cells)

		Data = OrderedData
		Points = MeasurementsPoints

	#Circular verification mesh
	else:
		#Compute mesh resolution and bounding radius
		resolution = np.mean(np.linalg.norm(Points - np.roll(Points,-1,axis=0),axis=1))
		radius = np.mean(np.linalg.norm(Points - np.average(Points,axis=1)))

		#Save mesh
		meshio.write_points_cells("robotMesh.mesh", Points, [("line",np.vstack((np.arange(len(Points)),np.roll(np.arange(len(Points)),-1))).T)])

	def processing_frequency(input):

		f = input[0]
		data = input[1]

		print("Processing frequency " + str(int(f)) + " Hz")

		#Define epslion for minimal distance between measurements
		if(elementType == "P0"):
			epsilon = meshResolution
		elif(elementType == "P1"):
			epsilon = meshResolution*1.5
		else:
			raise ValueError("Invalid element type")

		#Neighbours value filtering
		if(not meshPath is None and meshResolution < 0.1*np.sqrt(radius)):	 #Very arbitrary criterion
			filteredData = deepcopy(data)

			counter = 0
			for i,(point,pointData) in enumerate(zip(Points,data)):

				neighbours = np.where(np.linalg.norm(Points - point,axis=1) <= epsilon)[0]
				neighbours = np.delete(neighbours,np.where(neighbours == i))

				neighboursAbs = np.abs(data[neighbours])
				neighboursPhase = np.angle(data[neighbours])

				meanAbs = np.mean(neighboursAbs)
				meanPhase = np.mean(neighboursPhase)
				stdAbs = np.std(neighboursAbs)
				stdPhase = np.std(neighboursPhase)

				#TO TEST
				meanNeighbours = np.mean(data[neighbours])
				stdNeighbours = np.std(data[neighbours])
				#if(np.abs(pointData - meanNeighbours) > 3*stdNeighbours):

				if(np.abs(np.abs(pointData) - meanAbs) > 3*stdAbs or np.abs(np.angle(pointData) - meanPhase) > 3*stdPhase):
					counter += 1
					filteredData[i] = meanAbs*np.exp(1j*meanPhase)
					print("Replacing data at point " + str(point) + " : initial data = " + str(pointData) + " - filtered data = " + str(filteredData[i]))

			data = filteredData
			print("Percentage of filtered values : " + str(np.round(100*counter/len(Points),3)) + " %")
		else:
			print("Not enough points to filter data based on neighbour values")
					
		if(INTERACTIVE):
			axAmp = go.Figure()
			axPhase = go.Figure()
		else:
			figAmp,axAmp = plt.subplots(1,figsize=figsize,subplot_kw=dict(projection='3d'))
			figPhase,axPhase = plt.subplots(1,figsize=figsize,subplot_kw=dict(projection='3d'))

		plot_3d_data(np.abs(data), Points, axAmp, label = r"$|$H$|$ (Pa/V)", interactive=INTERACTIVE)
		plot_3d_data(wrap(np.angle(data)), Points, axPhase, label = "Phase (rad)", interactive=INTERACTIVE)

		if(pointCloudPath is not None):
			plotPointCloudFromPath(pointCloudPath, ax = axAmp, interactive=INTERACTIVE)
			plotPointCloudFromPath(pointCloudPath, ax = axPhase, interactive=INTERACTIVE)

		if(meshPath is not None):
			plotMesh(MeasurementsVertices, Faces, ax = axAmp, interactive=INTERACTIVE)
			plotMesh(MeasurementsVertices, Faces, ax = axPhase, interactive=INTERACTIVE)

		#set_title(axAmp,"Pressure/Input signal TFE amplitude at " + str(int(f)) + " Hz\n1/" + str(octBand) + " octave smoothing")
		#set_title(axPhase,"Pressure/Input signal TFE phase at " + str(int(f)) + " Hz\n1/" + str(octBand) + " octave smoothing")
		#axAmp.set_title("Measured amplitude at " + str(int(f)) + " Hz")
		#axPhase.set_title("Measured phase at " + str(int(f)) + " Hz")
			
		if(INTERACTIVE):
			save_fig(axAmp, folderName + "/amplitude_" + str(int(f)) + ".html",interactive=True)
			save_fig(axPhase, folderName + "/phase_" + str(int(f)) + ".html",interactive=True)
		else:
			save_fig(figAmp, folderName + "/amplitude_" + str(int(f)) + ".pdf")
			save_fig(figPhase, folderName + "/phase_" + str(int(f)) + ".pdf")
			plt.close("all")

		#Save data at given frequency
		np.savetxt(folderName + "/data_" + str(int(f)) + ".csv",np.array([np.real(data),np.imag(data)]).T,delimiter=",")

	with Pool(os.cpu_count()-1) as pool:
		pool.map(processing_frequency,list(zip(Frequencies,Data)))