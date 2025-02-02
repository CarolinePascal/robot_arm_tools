from setuptools import setup

setup(
   name='robot_arm_acoustic',
   version='0.0.0',
   description='The robot_arm_acoustic package',
   author='Caroline PASCAL',
   author_email='caroline.pascal.2020@ensta-paris.fr',
   packages=['robot_arm_acoustic',
             'robot_arm_acoustic.measurements',
             'robot_arm_acoustic.simulations'],  
   package_dir={'': 'post_processing',
                'robot_arm_acoustic.measurements': 'post_processing/robot_arm_acoustic',
                'robot_arm_acoustic.simulations': 'post_processing/robot_arm_acoustic'},
   install_requires=[
      #Arrays
      "numpy",
      #Plots
      "matplotlib",
      "plotly",
      #YAML file handling
      "pyyaml",
      #Mesh handling
      "antiprism_python",
      "scipy",
      "meshio",
      "dualmesh",
      "trimesh",
      #Point cloud handling
      "point_cloud_utils",
      "open3d",
      #CLI
      "cloup",
      #Acoustic packages
      "unyt",
      "csaps",
      "measpy",
   ]
)