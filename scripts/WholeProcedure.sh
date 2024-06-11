#! /bin/bash
#dir="/home/$(whoami)/AcousticMeasurements/05-06-2024/"
#dir="/media/demo1/Transcend/AcousticMeasurements/07-06-2024_0.02/"
#directory="/home/demo1/AcousticMeasurements/09-06-2024/"
directory="/home/demo1/AcousticMeasurements/11-06-2024/"

roslaunch robot_arm_acoustic ControlPoints.launch simulation:=false measurement_server_storage_folder:=$directory/ControlPoints/Init/ safety_distance:=0.04

#roslaunch robot_arm_acoustic MeshMeasurement.launch simulation:=false safety_distance:=0.04 measurement_server_storage_folder:=$directory/MeshMeasurements/ mesh_path:="/home/demo1/catkin_ws/src/robot_arm_acoustic/config/meshes/custom_0.05/MeasurementsMesh/P1/MeasurementsMesh.yaml" gradient:=true 
#roslaunch robot_arm_acoustic MeshMeasurement.launch simulation:=false safety_distance:=0.04 measurement_server_storage_folder:=$directory/MeshMeasurements/ mesh_path:="/home/demo1/catkin_ws/src/robot_arm_acoustic/config/meshes/custom_0.02/MeasurementsMesh/P1/MeasurementsMesh.yaml" gradient:=false 
roslaunch robot_arm_acoustic MeshMeasurement.launch simulation:=false safety_distance:=0.04 measurement_server_storage_folder:=$directory/MeshMeasurements/ mesh_path:="/home/demo1/catkin_ws/src/robot_arm_acoustic/config/meshes/custom_test/MeasurementsMesh/P1/MeasurementsMesh.yaml" gradient:=false

roslaunch robot_arm_acoustic DirectivityMeasurement.launch simulation:=false trajectory_radius:=0.20 trajectory_steps_number:=30 trajectory_center_pose:=[0.477,0.007,0.437] trajectory_axis:=[0.5,0.0,0.866] measurement_server_storage_folder:=$directory/Verification/Verification_gradient/ safety_distance:=0.04 gradient:=true gradient_offset:=0.005

roslaunch robot_arm_acoustic DirectivityMeasurement.launch simulation:=false trajectory_radius:=0.25 trajectory_steps_number:=20 trajectory_center_pose:=[0.477,0.007,0.437] trajectory_axis:=[0.5,0.0,0.866] measurement_server_storage_folder:=$directory/Verification/Verification_no_gradient/ safety_distance:=0.04 gradient:=false gradient_offset:=0.0

roslaunch robot_arm_acoustic ControlPoints.launch simulation:=false measurement_server_storage_folder:=$directory/ControlPoints/Final/ safety_distance:=0.04
