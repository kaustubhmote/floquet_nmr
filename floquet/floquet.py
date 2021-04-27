#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm


def F(n=None, fdim=5, hdim=None, term=None, symbolic=False):

    if not symbolic:
        return fmat_numeric(n=n, fdim=fdim, hdim=hdim, term=term)

    else:
        return fmat_symbolic(n=n, fdim=fdim, hdim=hdim, term=term)


def fmat_symbolic(n=None, fdim=5, hdim=None, term=None):
    """
    Symbolic Floquet Matrix Generator
    
    Parameters
    ----------
    n : int or None
        N in F_N
    dim : int
        dimension of the Floquet space
    hdim : int
        dimension of Hilbert space
    term : sym.Matrix
        term that goes into the matrix, defaults to 
        Id in case n
        
    Returns
    -------
    M : sym.Matrix
        Matrix of dimansions (2*dim+1)*hdim
    
    """
    import sympy as sym
    S = sym.S

    if term is None:
        term = S(1)
    
    half_fdim = S(int((fdim - 1)//2))
    
    if hdim is None:
        if term == S(1):
            hdim = S(1)
        else:
            hdim = S(2)
    else:
        hdim = S(int(hdim))
    
    if n is not None:
        n = S(n)
        abs_n = abs(S(n))
    else:
        abs_n = S(0)
    
    if abs_n >= fdim:
        raise ValueError("n has to be less than 2*dim+1")
    
    hpad = sym.ones(S(0), abs_n*hdim) 
    vpad = sym.ones(abs_n*hdim, S(0)) 
    
    
    if n is None:
        E = sym.diag(*[S(1)]*hdim)
        M = sym.diag(*[S(i)*term*E for i in range(half_fdim, -half_fdim-1, -1)])

    elif n > S(0):
        M = sym.diag(hpad, *[term]*(fdim-abs_n), vpad)
    
    elif n < S(0):
        M = sym.diag(vpad, *[term]*(fdim-abs_n), hpad)
        
    elif n == S(0):
        M = sym.diag(*[term]*(fdim))
                   
    return M



def fmat_numeric(n=None, fdim=5, hdim=None, term=None,):
    """
    Calculates the Floquet hamiltonian
    
    Parameters
    ----------
    
    n: int or tuple(int, int)
        Floquet Operator specification
        Defaults to the Number Operator 
    fdim: int
        dimension of the Floquet space
        Actual dimension is 2*fdim + 1
    hdim: int
        dimension of the Hilbert space
    term: np.ndarray or float
        term to be multiplied to the Floquet operator 
        or the Number operator 
        
    
    Returns
    -------
    matrix : np.ndarray
        Floquet matrix


    Notes
    -----
    This function looks like this for reasons of speed
    rather than readability. Currently ~75% of the time
    is spent creating the zero matrix. Maybe sparse
    representation will help?


    """
    # default to a single spin-half
    if hdim is None:
        hdim = 2

    # default to the number operator
    if n is None:
        absn = 0
        loc = None

    elif isinstance(n, int):
        loc = None    
        absn = int(np.abs(n))

    elif isinstance(n, list) or isinstance(n, tuple):
        loc = int(n[1] + fdim)
        n = int(n[0])
        absn = int(np.abs(n))

    else:
        raise ValueError(f"{n} is not an acceptable position")  


    # initilization
    mdim = (2 * fdim + 1) * hdim
    matrix = np.zeros((mdim, mdim), dtype="complex128")

    if n is None:
        if term is None:
            omega = 1.0
        else:
            try:
                omega = float(term)
            except:
                raise TypeError("For the number operator, the term must be a number (omega)")

    else:
        if term is None:
            # short circuit and give a blank matrix
            return matrix

        elif not isinstance(term, np.ndarray):
            try:
                term = float(term) * np.diag([1]*hdim)
            except:
                raise TypeError("term must be either a numpy array, a number, or None")




    # indices for the submatrix
    submatrices = list(range(0, mdim - (absn + 1) * hdim + 1, hdim))
    if loc is not None:
        submatrices = [submatrices[loc]]


    # put in the terms
    for s in submatrices:
        move = s + absn * hdim

        # defaults to the Number operator
        if n is None:
            matrix[s : s + hdim, s : s + hdim] = np.diag([-1]*hdim) * s / 2
            matrix[s : s + hdim, s : s + hdim] += np.diag([ (2 * fdim + 1) // 2 ] * hdim)
            matrix[s : s + hdim, s : s + hdim] *= omega

        elif n > 0:
            matrix[s : s + hdim, move : move + hdim] = term

        elif n < 0:
            matrix[move : move + hdim, s : s + hdim] = term

        elif n == 0:
            matrix[s : s + hdim, s : s + hdim] += term

    return matrix


def N(**kwargs):
    """
    Number Operator

    """

    return F(n=None, **kwargs)


def quinton(matrix, figax=None, **kwargs):
    """
    Quinton Plot for a given (square) matrix
    """

    # set the matplotlib axis, or make one if not given
    if figax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    else:
        fig, ax = figax

    # see if hilbert dimension is given, otherwise set to 2
    try:
        hdim = kwargs["hilbert_dim"]
    except:
        hdim = 2

    # Floquet matrix dimensions
    dim = matrix.shape

    # scaling
    if "norm" not in kwargs:
        if "vmax" not in kwargs.keys():
            vmax = np.max(np.abs(matrix).real)
        else:
            vmax = kwargs["vmax"]

        if "vmin" not in kwargs.keys():
            vmin = -vmax
        else:
            vmax = kwargs["vmin"]

        img = ax.imshow(
            matrix.real, interpolation=None, vmax=vmax, vmin=vmin, cmap="coolwarm"
        )
    else:
        img = ax.imshow(
            matrix.real, interpolation=None, norm=kwargs["norm"], cmap="coolwarm"
        )

    ax.axis("equal")

    # make pretty
    for s in ax.set_xticks, ax.set_yticks:
        s([i - 0.5 for i in range(0, matrix.shape[0] - 1, hdim)])
    for s in ax.set_xticklabels, ax.set_yticklabels:
        s([])
    ax.tick_params(**{pos: False for pos in ["top", "left", "right", "bottom"]})
    ax.set_frame_on(False)

    ax.hlines(-0.5, -0.5, dim[1] - 0.5, lw=1)
    ax.hlines(dim[0] - 0.49, -0.5, dim[1] - 0.5, lw=1)
    ax.vlines(-0.5, -0.5, dim[1] - 0.5, lw=1)
    ax.vlines(dim[0] - 0.49, -0.5, dim[1] - 0.5, lw=1)
    ax.grid(color="w", linestyle="-", linewidth=1)
    ax.set_clip_on(True)

    if "cbar" in kwargs.keys():
        if kwargs["cbar"]:
            cax = fig.add_axes([0.90, 0.24, 0.05, 0.6])
            fig.colorbar(img, cax=cax)

    return fig, ax
