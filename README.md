# Floquet Theory in Magnetic Resonance

[doi: 10.5281/zenodo.4727633](https://dx.doi.org/10.5281/zenodo.4727633)


This repository is intended as a pedagogical accompaniment to the following article: 
"Floquet Theory in Magnetic Resonance: Formalism and Applications" by 
Konstantin L. Ivanov, Kaustubh R. Mote, Matthias Ernst, Asif Equbal, and P. K. Madhu 
appearing in Progress in Nuclear Magnetic Resonance Spectroscopy (2021).

The `floquet` module provides some convenience functions to generate Floquet Hamiltonian (single-mode), and carry out rotations. The actual calculations for cases such as the analysis of MAS sidebands are collected as Jupyter Notebooks in the `examples` directory. 


# Requirements
1. Python 3.7 or higher
1. Numpy
1. Matplotlib 
1. Sympy (optional, required for generating symbolic hamiltonians)
1. NMRGlue (optional, required for reading in data generated by SIMPSON )
1. SIMPSON (optional, required for running comparisons with Floquet)


# Installation

1. Using pip (and git)

```
[activate your virtual environment]
python -m pip install git+https://github.com/kaustubhmote/floquet_nmr
```

2. Install from source
```
git clone https://github.com/kaustubhmote/floquet_nmr
cd pulseplot
[activate your virtual environment]
python -m pip install -r requirements.txt
python -m pip install .
```

## Usage

```python

>>> from floquet import F, N, pauli
>>> I = pauli()
>>> hamiltonian = F(n=-1, fdim=1, term=I["x"]) + N(fdim=1, term=10)
>>> print(hamiltonian)

array([[ 10. +0.j,   0. +0.j,   0. +0.j,   0. +0.j,   0. +0.j,   0. +0.j],
       [  0. +0.j,  10. +0.j,   0. +0.j,   0. +0.j,   0. +0.j,   0. +0.j],
       [  0. +0.j,   0.5+0.j,   0. +0.j,   0. +0.j,   0. +0.j,   0. +0.j],
       [  0.5+0.j,   0. +0.j,   0. +0.j,   0. +0.j,   0. +0.j,   0. +0.j],
       [  0. +0.j,   0. +0.j,   0. +0.j,   0.5+0.j, -10. +0.j,   0. +0.j],
       [  0. +0.j,   0. +0.j,   0.5+0.j,   0. +0.j,   0. +0.j, -10. +0.j]])

```

For examples on how to use these hamiltonians in calculations, please see below:

## Examples

1. [Calculation of MAS Sidebands](examples/MAS_Sidebands.ipynb "sidebands")

Describes how a basic MAS calculation can be done using the Floquet formalism.

2. [Floquet Detection Operator](examples/Floquet_Detection_Operator.ipynb "detect")

This describes how different detection operators give rise to different truncation.
artefacts.

3. [Comparison with numerical calculations in Hilbert space](examples/Floquet_Simpson_Comparison.ipynb "compare")

A comparison between calculations done using the SIMPSON programme and Floquet Formalism.
This notebook requires SIMPSON to be installed and available in the PATH for complete
execution.

4. [Analysis of decoupling sequences](examples/Operator_based_Floquet_XiX.ipynb "xix")

This notebook describes how the Operator-based Floquet Theory can be leveraged to understand
essential features of decoupling sequences using the XiX decoupling sequence as an example.
