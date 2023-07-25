#! /bin/bash

#TODO Script parameters ? 

output_dir="/home/$(whoami)/AcousticMeasurements/"

studied_object_file="/home/demo1/Desktop/robot_arm_acoustic/config/environments/CollisionVolumes.yaml"

mesh_size=0.3
mesh_resolution=0.02

verification_radius=0.25
verification_steps=20

gradient_steps=5
gradient_steps_size=0.01

#YAML parser
function parse_yaml {
   local prefix=$2
   local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
   sed -ne "s|,$s\]$s\$|]|" \
        -e ":1;s|^\($s\)\($w\)$s:$s\[$s\(.*\)$s,$s\(.*\)$s\]|\1\2: [\3]\n\1  - \4|;t1" \
        -e "s|^\($s\)\($w\)$s:$s\[$s\(.*\)$s\]|\1\2:\n\1  - \3|;p" $1 | \
   sed -ne "s|,$s}$s\$|}|" \
        -e ":1;s|^\($s\)-$s{$s\(.*\)$s,$s\($w\)$s:$s\(.*\)$s}|\1- {\2}\n\1  \3: \4|;t1" \
        -e    "s|^\($s\)-$s{$s\(.*\)$s}|\1-\n\1  \2|;p" | \
   sed -ne "s|^\($s\):|\1|" \
        -e "s|^\($s\)-$s[\"']\(.*\)[\"']$s\$|\1$fs$fs\2|p" \
        -e "s|^\($s\)-$s\(.*\)$s\$|\1$fs$fs\2|p" \
        -e "s|^\($s\)\($w\)$s:$s[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
        -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p" | \
   awk -F$fs '{
      indent = length($1)/2;
      vname[indent] = $2;
      for (i in vname) {if (i > indent) {delete vname[i]; idx[i]=0}}
      if(length($2)== 0){  vname[indent]= ++idx[indent] };
      if (length($3) > 0) {
         split($3,tab,"#");
         gsub(/ /,"",tab[1])
         vn=""; for (i=0; i<indent; i++) { vn=(vn)(vname[i])("_")}
         printf("%s%s%s=\"%s\"\n", "'$prefix'",vn, vname[indent], tab[1]);
      }
   }'
}

function decimal_addition {
    echo $(echo "$1 + $2" | bc -l | sed -E -e 's!(\.[0-9]*[1-9])0*$!\1!' -e 's!(\.0*)$!!')
}

function decimal_division {
    echo $(echo "$1 / $2" | bc -l | sed -E -e 's!(\.[0-9]*[1-9])0*$!\1!' -e 's!(\.0*)$!!')
}

function decimal_multiplication {
    echo $(echo "$1 * $2" | bc -l | sed -E -e 's!(\.[0-9]*[1-9])0*$!\1!' -e 's!(\.0*)$!!')
}

eval $(parse_yaml $studied_object_file)

#Initial control points
roslaunch robot_arm_acoustic ControlPoints.launch simulation:=false measurement_server_storage_folder:="${output_dir}AcousticMeasurements/ControlPoints/Init/"

#Mesh measurements
roslaunch robot_arm_acoustic MeshMeasurement.launch simulation:=false mesh_type:=sphere mesh_size:=$mesh_size mesh_resolution:=0.02 measurement_server_storage_folder:="${output_dir}AcousticMeasurements/MeshMeasurements/"

#Verification measurements : circular measurements
roslaunch robot_arm_acoustic DirectivityMeasurement.launch simulation:=false trajectory_radius:=0.25 trajectory_steps_number:=20 trajectory_center_pose:=[$objectPose_1,$objectPose_2,$objectPose_3,0,0,0] measurement_server_storage_folder:="${output_dir}AcousticMeasurements/Verification/Verification/"

#Gradient measurements
delta_gradient=$(decimal_multiplication $(decimal_division $(decimal_addition ${gradient_steps} -1) 2.0) ${gradient_steps_size})

#Left gradient measurements
roslaunch robot_arm_acoustic StraightMeasurement.launch simulation:=false trajectory_steps_number:=5 trajectory_axis:=[0,1,0] trajectory_step_size:=0.01 trajectory_start_pose:=[$objectPose_1,$(decimal_addition $(decimal_addition $objectPose_2 $(decimal_division $mesh_size 2.0)) -$delta_gradient),$objectPose_3,1.5708,0,0] measurement_server_storage_folder:="${output_dir}AcousticMeasurements/Verification/VerificationGradientL/"

#Right gradient measurements
roslaunch robot_arm_acoustic StraightMeasurement.launch simulation:=false trajectory_steps_number:=5 trajectory_axis:=[0,-1,0] trajectory_step_size:=0.01 trajectory_start_pose:=[$objectPose_1,$(decimal_addition $(decimal_addition $objectPose_2 -$(decimal_division $mesh_size 2.0)) $delta_gradient),$objectPose_3,-1.5708,0,0] measurement_server_storage_folder:="${output_dir}AcousticMeasurements/Verification/VerificationGradientR/"

#Final control points
roslaunch robot_arm_acoustic ControlPoints.launch simulation:=false measurement_server_storage_folder:="${output_dir}AcousticMeasurements/ControlPoints/Final/"
