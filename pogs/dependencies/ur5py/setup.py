"""
Setup of ur5py.
Author: Justin Kerr justin_kerr@berkeley.edu
"""
from setuptools import setup

setup(name='ur5py',
      version='0.1.0',
      description='UR5 library for the Berkeley Autolab which handles control, planning, kinematics', 
      author='Justin Kerr',
      author_email='justin_kerr@berkeley.edu',
      package_dir = {'': '.'},
      packages=['ur5py'],
     )
