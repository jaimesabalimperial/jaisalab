# setup.py
from setuptools import setup, Extension, find_packages

setup(name='thesis',
      version='0.0.1',
      # Specify packages (directories with __init__.py) to install.
      # You could use find_packages(exclude=['modules']) as well
      packages=['algos, experiments, envs'],
      include_package_data=True,
      )