# setup.py
from setuptools import setup, Extension, find_packages

setup(name='thesis',
      version='0.0.1',
      # Specify packages (directories with __init__.py) to install.
      # You could use find_packages(exclude=['modules']) as well
      packages=['algos, experiments, envs'],
      include_package_data=True,
      )
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="jaisalab",
    version="0.0.1",
    author="Jaime Sabal",
    author_email="jsabalb@gmail.com",
    description="RL Lab for Jaime Sabal's 2022 MSc AI Individual Project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jaimesabalimperial/thesis",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "jaisalab"},
    packages=setuptools.find_packages(where="jaisalab"),
    python_requires=">=3.6",
)