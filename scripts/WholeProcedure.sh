#! /bin/bash

roslaunch robot_arm_acoustic ControlPoints.launch simulation:=false measurement_server_storage_folder:=/home/demo1/Desktop/AcousticMeasurements/ControlPoints/Init/
roslaunch robot_arm_acoustic MeshMeasurement.launch simulation:=false mesh_type:=sphere mesh_size:=0.3 mesh_resolution:=0.02
roslaunch robot_arm_acoustic DirectivityMeasurement.launch simulation:=false trajectory_radius:=0.25 trajectory_steps_number:=20 trajectory_center_pose:=[0.43520395193183264,-0.015475821997893293,0.5020558723686521,0,0,0]
roslaunch robot_arm_acoustic StraightMeasurement.launch simulation:=false trajectory_steps_number:=5 trajectory_axis:=[0,1,0] trajectory_step_size:=0.01 trajectory_start_pose:=[0.43520395193183264,0.1145241780021067,0.5020558723686521,1.5708,0,0] measurement_server_storage_folder:=/home/demo1/Desktop/AcousticMeasurements/Verification/VerificationGradientL
roslaunch robot_arm_acoustic StraightMeasurement.launch simulation:=false trajectory_steps_number:=5 trajectory_axis:=[0,-1,0] trajectory_step_size:=0.01 trajectory_start_pose:=[0.43520395193183264,-0.1454758219978933,0.5020558723686521,-1.5708,0,0] measurement_server_storage_folder:=/home/demo1/Desktop/AcousticMeasurements/Verification/VerificationGradientR
roslaunch robot_arm_acoustic ControlPoints.launch simulation:=false measurement_server_storage_folder:=/home/demo1/Desktop/AcousticMeasurements/ControlPoints/Final/