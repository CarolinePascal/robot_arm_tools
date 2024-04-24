# robot_arm_acoustic

A Boundary Element Method (BEM) based Sound Field Estimation(SFE) tool based on robotized acoustic measurements.

## Intro

This package provides a set of tools to perform acoustic measurements using a robotic arm equiped with a microphone. 

It depends on the [robot_arm_tools](https://gitlab.ensta.fr/pascal.2020/robot_arm_tools) package, which offers a simplified interface for the use of robotic arms and the [MoveIt](https://moveit.ros.org) motion planning framework . It is compatible with most [serial robots included in MoveIt](https://moveit.ros.org/robots/). In its current version, the package was tested using ROS melodic, but shoul also work with ROS noetic as well.

The measurements are handled using the [measpy](https://github.com/odoare/measpy) python module, which is compatible with most sound cards and DACs.

This package also proposes a BEM-based SFE tool, which predicts the sound pressure radiated by any unknwon source, given a set of measurements located on a surrounding mesh. This tool is built using the BEM library implemented in [FreeFEM++](https://doc.freefem.org/documentation/BEM.html).

## Getting started

### Requirements 

#### Robotized measurements

* [ROS Melodic](http://wiki.ros.org/melodic/Installation/Ubuntu) or [ROS Noetic](http://wiki.ros.org/noetic/Installation/Ubuntu)

##### Installation from source

First, use git to clone the required packages into your catkin workspace : 

```bash
mkdir -p ~/catkin_ws/src && cd catkin_ws/src
git clone https://gitlab.ensta.fr/pascal.2020/robot_arm_tools
git clone https://gitlab.ensta.fr/pascal.2020/robot_arm_acoustic
```

> :warning: _Remark_ : In order to ensure accurate measurements, we advise to perform a preliminary calibration of the robot using dedicated tools, such as the [robot_arm_calibration](https://gitlab.ensta.fr/pascal.2020/robot_arm_calibration) ROS package.

Install dependencies : 

```bash
cd ~/catkin_ws
sudo apt update
rosdep update
rosdep install --from-paths src --ignore-src -y

cd src/robot_arm_acoustic
python3 -m pip install -e .
```

And build packages :

```bash
catkin build
source devel/setup.bash
```

> :warning: _Remark_ : before use, attention must be paid to the proper functioning of the _robot\_arm\_tool_ package, which involve the installation of the robot ROS driver !

#### SFE Tool

* [FreeFEM++](https://doc.freefem.org/introduction/installation.html)

## Usage

### Robotized measurements

#### Preliminary configurations

Before use, a few hardware related configuration files are to be created : 
1. *Robot configuration using `robot_arm_tools`*;
2. *Robot cell description* : A YAML file containing the description of the robot surroundings must be provided in `/config/environments/Environment.yaml` (the robot base frame is considered as the reference frame);
3. *Tool description* : The tool geometry must be defined according to the procedure described in the `robot_arm_tools` package. In practice, it only requires a CAD file describing the tool, and the exact position and location of the sensing point (which will ba labelled as the robot end-effector);
4. *Measurement script customization* : the required measurements will vary depending on thte targeted task and used sensors. The desired measurement scheme must be precised in the `src/SoundMeasurementServer.py` script, and fit `measpy` requirements (**v.0.0.15**);
5. *(Optionnal) Studied object description* : The geometric description of the studied object can be provided in `/config/environments/StudiedObject.yaml`.

#### Usage sample

For instance, a simple directivity measurement routine can be launched using : 

```bash
roslaunch robot_arm_acoustic robot_name:=<robot_name> tool_name:=<tool_name> simulation:=<true/false> trajectory_radius:=<radius> trajectory_steps_number:=<steps> trajectory_axis:=<[x,y,z]> trajectory_center_pose:=<[x,y,z,rx,ry,rz]>
```

See the other launch files under `/config/launch/` for more examples.

### SFE tools

The core of the BEM based SFE tool is implemented in `post_processing\robot_arm_acoustic\AcousticComputationBEM.edp`, and offers both single layer potential and combined layer potential indirect resolutions.

#### Simulation 

```bash
ffmpi-run -np <n_processors> AcousticComputationBEM.edp -wg -realMeasurements 0 -frequency <f> -size <D> -resolution <h> -dipoleDistance <a> -sigmaPosition <\sigma_P(m)> -sigmaMeasure <sigma_M(%)> + -fileID  <id> -verificationSize <D> -verificationResolution <h> -studiedFunction <function> -DelementType=<P0/P1> -Dgradient=<0/1> -ns
```

For the simulation to work properly, the computation and reconstruction meshes must already be generated calling the meshing tool `python3 robot_arm_acoustic.MeshTools` with the `--save_folder` option set to `./meshes/`. **By default, the computation mesh is a spheric mesh (P0/P1) and the reconstruction mesh is a circular mesh (P1).**

_Remark_ : An **much** easier way to use the simulation is to run the sample python script :
```bash
python3 -m robot_arm_acoustic.simulations.RunSimulationsAcousticComputation
```

#### Real measurements

```bash
ffmpi-run -np <n_processors> AcousticComputationBEM.edp -wg -realMeasurements 1 -frequency <f> -measurementsMeshPath <path> -measurementsDataPath <data> -verificationMeshPath <path> -verificationDataPath <path> -verificationGradientDataFolder <path> -DelementType=<P0/P1> -Dgradient=<0/1> -ns"

```

Meshes must be provided as `.mesh` files, and correctly fit with the chosen elements (P0/P1) and measurements locations. For instance, with P0 elements, measurements must be performed on the faces centroids !

Data must be provided in a `.csv` file, and represent the complex measured data for the chosen frequency. Each row of the file contains the measured real part and imaginary part, and the rows must be ordered according to the mesh data (i.e. according to the faces for P0 elements, and according to the vertices for P1 elements).

_Remark_ : Run (and read) the sample python file for more information, which can directly process data obtained with our pre-processing tools :
```bash
python3 -m robot_arm_acoustic.measurements.RunMeasurementsAcousticComputation
```

##### Measurements pre-processing tools

Using measurements generated by `measpy`, a simple data processing tool is available via the `robot_arm_acoustic` package. **For now this tool only works with spheric meshes**.

```bash
python3 -m robot_arm_acoustic.DataProcessingMeasurements <TFE_method:welch/farina> <input_type:sweep/noise> <input_ID> <output_id> <frequencies (coma separated values)> <point_cloud_path (optionnal)> <mesh_path> <element_type:P0/P1>
```

This script is to be run in the folder where the measurements are stored (as `.csv` and `.wav` files), alongside with the measurements mesh. It will generate a folder containing the correctly ordered and completed (missing measurements) data files, ready for the SFE procedure.