#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.md") as readme_file:
    readme = readme_file.read()

requirements = [
    "numpy",
    "scipy",
    "matplotlib",
]

setup(
    name="floquet_nmr",
    description="A module to demonstrate the use of Floquet Theory in NMR",
    version="0.1",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Kaustubh R. Mote",
    python_requires=">=3.7",
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
    ],
    install_requires=requirements,
    include_package_data=True,
    packages=find_packages(include=["floquet"]),
    setup_requires=requirements,
    url="https://github.com/kaustubhmote/floquet_nmr",
    zip_safe=False,
)
