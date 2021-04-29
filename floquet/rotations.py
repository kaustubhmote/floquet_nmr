from functools import lru_cache

import numpy as np
from numpy import cos, exp, pi, sin, sqrt

def identity(dim):
    """
    Returns an identity matrix with a given dimension

    """
    return np.diag(np.ones(dim, dtype="complex128"))


def pauli(normalized=True):

    if normalized:
        c = 0.5
    else:
        c = 1.0

    I = {}
    I["x"] = c * np.array([[0, 1], [1, 0]], dtype="complex128")
    I["y"] = c * np.array([[0, -1j], [1j, 0]], dtype="complex128")
    I["z"] = c * np.array([[1, 0], [0, -1]], dtype="complex128")
    I["p"] = I["x"] + 1j * I["y"]
    I["m"] = I["x"] - 1j * I["y"]

    return I


def tensor_setup(sigmax, sigmay, sigmaz):
    """ Constructs a tensor from the three values given"""

    return np.diag((sigmax, sigmay, sigmaz))


def tensor_setup2(aniso, asym, iso=0):
    """ Constructs a tensor from the anisotropy and asymmetry values given"""

    values = -0.5 * (1 + asym), -0.5 * (1 - asym), 1
    sigmax, sigmay, sigmaz = [i * aniso + iso for i in values] 

    return np.diag((sigmax, sigmay, sigmaz))


def cartesian_tensor_basis():
    """ Cartesian Tensor basis """

    T = {}
    for i, n1 in enumerate(("x", "y", "z")):
        for j, n2 in enumerate(("x", "y", "z")):
            Tbase = np.zeros((3, 3,), dtype="complex128")
            Tbase[i, j] = 1
            T[f"{n1}{n2}"] = Tbase

    return T


def spherical_tensor_basis(dtype="space", coord=None):
    """ Spherical Tensor Basis """

    if dtype == "space":
        T = cartesian_tensor_basis()
        coord = None

    elif dtype == "spin":
        I = pauli()
        spin = [ I["x"], I["y"], I["z"], ]

        T = {}
        for c, i in zip(coord, ("x", "y", "z")):
            for s, j in zip(spin, ("x", "y", "z")):
                T[f"{i}{j}"] = s * c

    SpT = {}
    SpT[(0, 0)] = -1.0 * sqrt(1 / 3.0) * (T["xx"] + T["yy"] + T["zz"])
    SpT[(1, 0)] = -1j * sqrt(0.5) * (T["xy"] - T["yx"])
    SpT[(1, 1)] = -0.5 * (T["zx"] - T["xz"] + 1j * (T["zy"] - T["yz"]))
    SpT[(1, -1)] = -0.5 * (T["zx"] - T["xz"] - 1j * (T["zy"] - T["yz"]))
    SpT[(2, 0)] = sqrt(1 / 6) * (3 * T["zz"] - (T["xx"] + T["yy"] + T["zz"]))
    SpT[(2, 1)] = -0.5 * (T["xz"] + T["zx"] + 1j * (T["yz"] + T["zy"]))
    SpT[(2, -1)] = 0.5 * (T["xz"] + T["zx"] - 1j * (T["yz"] + T["zy"]))
    SpT[(2, 2)] = 0.5 * (T["xx"] - T["yy"] + 1j * (T["xy"] + T["yx"]))
    SpT[(2, -2)] = 0.5 * (T["xx"] - T["yy"] - 1j * (T["xy"] + T["yx"]))

    return SpT


def matrix_to_sphten(tensor):
    """ give coefs from tensor """

    T = spherical_tensor_basis()
    coeff = {}
    for k, v in T.items():
        coeff[k] = np.trace(v.conj().T @ tensor)

    return coeff


def sphten_to_matrix(coeffs):
    """ give tensor from coeffs """

    T = spherical_tensor_basis()
    t = np.zeros((3, 3), dtype="complex128")

    for k in range(3):
        for j in range(-k, k + 1):
            t += coeffs[k, j] * T[k, j]

    return t


@lru_cache(maxsize=128)
def wigner_d_beta(beta):
    """
    Wigner D elements copy pasted from the output of following code:

    from sympy.physics.quantum.spin import Rotation as Rot
    from sympy import symbols

    b = symbols("beta")
    for i in range(3):
        for j in range(-i, i+1):
            for k in range(-i, i+1):
                print(f"d[{i}][{j}, {k}] =", Rot.d(i, j, k, b).doit())

    """

    d = {}
    d[0], d[1], d[2] = {}, {}, {}

    d[0][0, 0] = 1.0
    d[1][-1, -1] = cos(beta) / 2 + 1 / 2
    d[1][-1, 0] = sqrt(2) * sin(beta) / 2
    d[1][-1, 1] = 1 / 2 - cos(beta) / 2
    d[1][0, -1] = -sqrt(2) * sin(beta) / 2
    d[1][0, 0] = cos(beta)
    d[1][0, 1] = sqrt(2) * sin(beta) / 2
    d[1][1, -1] = 1 / 2 - cos(beta) / 2
    d[1][1, 0] = -sqrt(2) * sin(beta) / 2
    d[1][1, 1] = cos(beta) / 2 + 1 / 2
    d[2][-2, -2] = cos(beta) / 2 + cos(2 * beta) / 8 + 3 / 8
    d[2][-2, -1] = (cos(beta) + 1) * sin(beta) / 2
    d[2][-2, 0] = sqrt(6) * sin(beta) ** 2 / 4
    d[2][-2, 1] = sin(beta) / 2 - sin(2 * beta) / 4
    d[2][-2, 2] = -cos(beta) / 2 + cos(2 * beta) / 8 + 3 / 8
    d[2][-1, -2] = -(cos(beta) + 1) * sin(beta) / 2
    d[2][-1, -1] = cos(beta) / 2 + cos(2 * beta) / 2
    d[2][-1, 0] = sqrt(6) * sin(2 * beta) / 4
    d[2][-1, 1] = cos(beta) / 2 - cos(2 * beta) / 2
    d[2][-1, 2] = sin(beta) / 2 - sin(2 * beta) / 4
    d[2][0, -2] = sqrt(6) * sin(beta) ** 2 / 4
    d[2][0, -1] = -sqrt(6) * sin(2 * beta) / 4
    d[2][0, 0] = 3 * cos(2 * beta) / 4 + 1 / 4
    d[2][0, 1] = sqrt(6) * sin(2 * beta) / 4
    d[2][0, 2] = sqrt(6) * sin(beta) ** 2 / 4
    d[2][1, -2] = (cos(beta) - 1) * sin(beta) / 2
    d[2][1, -1] = cos(beta) / 2 - cos(2 * beta) / 2
    d[2][1, 0] = -sqrt(6) * sin(2 * beta) / 4
    d[2][1, 1] = cos(beta) / 2 + cos(2 * beta) / 2
    d[2][1, 2] = (cos(beta) + 1) * sin(beta) / 2
    d[2][2, -2] = -cos(beta) / 2 + cos(2 * beta) / 8 + 3 / 8
    d[2][2, -1] = (cos(beta) - 1) * sin(beta) / 2
    d[2][2, 0] = sqrt(6) * sin(beta) ** 2 / 4
    d[2][2, 1] = -(cos(beta) + 1) * sin(beta) / 2
    d[2][2, 2] = cos(beta) / 2 + cos(2 * beta) / 8 + 3 / 8

    return d


def wignerd(alpha, beta, gamma):
    """
    Wigner D matrix elements for each
    """

    # get the beta terms
    d = wigner_d_beta(beta)

    # put in alpha and gamma terms
    D = {}
    D[0], D[1], D[2] = {}, {}, {}

    # TODO: either numba jit-compile or cast to matrix-form 
    for i in range(3):
        for k, v in d[i].items():
            D[i][k] = exp(-1j * alpha * k[0]) * exp(-1j * gamma * k[1]) * v

    return D


def _rotate_single(ca, label=None, angle=None):
    """ rotate around a single axis """

    angles = {"alpha": 0, "beta": 0, "gamma": 0}
    
    if label in angles.keys():
        angles[label] = angle
    else:
        raise ValueError("label must be 'alpha', 'beta' or 'gamma'")
        
    
    cb = {}
    D = wignerd(**angles)

    # TODO: either numba jit-compile or cast to matrix-form 
    for k in range(3):
        for q in range(-k, k + 1):
            cb[k, q] = sum( [ D[k][q, j] * ca[k, j]  for j in range(-k, k + 1) ] )

    return cb


def rotate(ca, alpha, beta, gamma):
    """ rotate """

    cb = _rotate_single(ca, "alpha", alpha)
    cb = _rotate_single(cb, "beta", beta)
    cb = _rotate_single(cb, "gamma", gamma)

    return cb
