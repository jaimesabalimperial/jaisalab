# setup.py
import setuptools 


with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="jaisalab",
    version="0.0.1",
    author="Jaime Sabal Bermúdez",
    author_email="jsabalb@gmail.com",
    description="Framework that builds on the garage toolkit for the evaluation and \
                 development of constrained RL algorithms.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jaimesabalimperial/jaisalab",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "jaisalab"},
    packages=setuptools.find_packages(where="jaisalab"),
    python_requires=">=3.6",
)