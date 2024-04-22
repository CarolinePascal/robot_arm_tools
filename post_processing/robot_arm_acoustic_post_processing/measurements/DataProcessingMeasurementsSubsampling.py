#Utility packages
import numpy as np
import glob
import os
import sys
import csv

#Mesh packages
import meshio as meshio
import trimesh as trimesh

#Plot package
import matplotlib.pyplot as plt
import plotly.graph_objects as go

#Acoustic tools
from measpy._tools import wrap

#Multiprocessing package
from multiprocessing import Pool

from robot_arm_acoustic_post_processing.measurements.PlotTools import plot_3d_data, save_fig, figsize
from robot_arm_acoustic_post_processing.measurements.DataProcessingMeasurements import sphereFit

from robot_arm_acoustic.MeshTools import plotMesh, plotPointCloudFromPath

INTERACTIVE = False

if __name__ == "__main__":

	#Get mesh path
	meshPath = ""
	try:
		meshPath = sys.argv[1]
		if(not os.path.isfile(meshPath)):
			raise(IndexError)
	except IndexError:
		print("Invalid mesh path !")
		sys.exit(-1)

	#Get mesh element type
	elementType = "P0"
	try:
		elementType = sys.argv[2]	
	except IndexError:
		print("Invalid element type, defaulting to " + elementType)

	#Get point cloud path
	pointCloudPath = None
	try:
		pointCloudPath = sys.argv[3]
	except IndexError:
		pass
	if(not pointCloudPath is None and not os.path.isfile(pointCloudPath)):
		pointCloudPath = None

	#Get initial measurements points
	initElementType = os.getcwd().split("_")[-1]

	initMesh = meshio.read("robotMesh.mesh")
	initVertices, initFaces = initMesh.points, initMesh.get_cells_type("triangle")

	if(initElementType == "P0"):
		initMeasurementsPoints = np.mean(initVertices[initFaces],axis=1)
	elif(initElementType == "P1"):
		initMeasurementsPoints = initVertices
	else:
		raise ValueError("Invalid element type")

	###TO ADAPT DEPENDING ON MESH###
	center,radius = sphereFit(initMeasurementsPoints)
	################################
	
	initResolution = trimesh.Trimesh(initVertices, initFaces).edges_unique_length.max()

	#Get new measurements points
	newMesh = meshio.read(meshPath)
	newVertices, newFaces = newMesh.points, newMesh.get_cells_type("triangle")
	newVertices += center

	if(elementType == "P0"):
		newMeasurementsPoints = np.mean(newVertices[newFaces],axis=1)
	elif(elementType == "P1"):
		newMeasurementsPoints = newVertices
	else:
		raise ValueError("Invalid element type")

	newResolution = float(trimesh.Trimesh(newVertices, newFaces).edges_unique_length.max())

	#Compute initial-new measurements points correspondances
	correspondances = []
	for newPoint in newMeasurementsPoints:
		distances = np.linalg.norm(initMeasurementsPoints - newPoint,axis=1)
		index = np.argmin(distances)
		if(distances[index] > initResolution/2):
			print("Invalid sub-sampled mesh !")
			sys.exit(-1)
		correspondances.append(index)

	#Get inital measurements data
	Files = sorted(glob.glob("data*.csv"), key=lambda file:int(os.path.basename(file).split(".")[0].split("_")[-1]))

	#Sub-sample initial measuements 
	folderName = os.getcwd() + "/" + str(np.round(newResolution,3))
	os.makedirs(folderName,exist_ok=True)

	def processing_file(file):
		print("Data processing file : " + file)

		f = int(os.path.basename(file).split(".")[0].split("_")[-1])

		with open(file, 'r') as csvfile:
			reader = csv.reader(csvfile, delimiter=',')
			Data = [float(row[0]) + 1j*float(row[1]) for row in reader]
		Data = np.array(Data)[correspondances]
					
		if(INTERACTIVE):
			axAmp = go.Figure()
			axPhase = go.Figure()
		else:
			figAmp,axAmp = plt.subplots(1,figsize=figsize,subplot_kw=dict(projection='3d'))
			figPhase,axPhase = plt.subplots(1,figsize=figsize,subplot_kw=dict(projection='3d'))

		plot_3d_data(np.abs(Data), newMeasurementsPoints, axAmp, label = r"$|$H$|$ (Pa/V)", interactive=INTERACTIVE)
		plot_3d_data(wrap(np.angle(Data)), newMeasurementsPoints, axPhase, label = "Phase (rad)", interactive=INTERACTIVE)

		if(pointCloudPath is not None):
			plotPointCloudFromPath(pointCloudPath, ax = axAmp, interactive=INTERACTIVE)
			plotPointCloudFromPath(pointCloudPath, ax = axPhase, interactive=INTERACTIVE)

		if(meshPath is not None):
			plotMesh(newVertices, newFaces, ax = axAmp, interactive=INTERACTIVE)
			plotMesh(newVertices, newFaces, ax = axPhase, interactive=INTERACTIVE)

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
		newfile = folderName + "/" + os.path.basename(file)
		np.savetxt(newfile,np.array([np.real(Data),np.imag(Data)]).T,delimiter=",")

	with Pool(os.cpu_count()-1) as pool:
		pool.map(processing_file,Files)

	#Save sub-sampled mesh
	meshio.write_points_cells(folderName + "/robotMesh.mesh", newVertices, newMesh.cells)







