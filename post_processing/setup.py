from setuptools import setup

setup(
   name='robot_arm_acoustic_post_processing',
   version='0.0.0',
   description='The robot_arm_acoustic_post_processing package',
   author='Caroline PASCAL',
   author_email='caroline.pascal.2020@ensta-paris.fr',
   packages=['robot_arm_acoustic_post_processing',
             'robot_arm_acoustic_post_processing.measurements',
             'robot_arm_acoustic_post_processing.simulations'], 
   package_dir={'robot_arm_acoustic_post_processing': '.',
                'robot_arm_acoustic_post_processing.measurements': './robot_arm_acoustic_post_processing',
                'robot_arm_acoustic_post_processing.simulations': './robot_arm_acoustic_post_processing'},
   install_requires=[
      "measpy",
      "numpy",
      "matplotlib",
      "plotly",
      "meshio",
      "trimesh",
      "open3d",
      "unyt",
      "csaps",
      "cloup",
   ]
)