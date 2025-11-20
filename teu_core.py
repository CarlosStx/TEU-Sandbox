# teu_core.py
"""
Core functions for the TEU Sandbox.
Two–scale structural vacuum rigidity model:
    lambda(r) = lambda0 / (1 + r/L1 + (r/L2)^2)
and effective TEU acceleration:
    a_TEU(r) = a_N(r) * [ lambda(r0) / lambda(r) ].
"""

import numpy as np

# Physical constants
G = 6.674e-11           # Gravitational constant (SI)
Msun = 1.989e30         # Solar mass (kg)
a0 = 1.2e-10            # MOND scale (m/s^2), for reference


def lambda_two_scale(r, lambda0, L1, L2):
    """
    Two–scale vacuum rigidity profile.

    Parameters
    ----------
    r : array_like
        Radius (m).
    lambda0 : float
        Reference rigidity (dimensionless).
    L1 : float
        Galactic scale (m).
    L2 : float
        Cosmological scale (m).

    Returns
    -------
    lam : ndarray
        Rigidity lambda(r).
    """
    r = np.asarray(r, dtype=float)
    return lambda0 / (1.0 + (r / L1) + (r / L2) ** 2)


def a_newton(r, M):
    """
    Newtonian radial acceleration for a point mass M.

    a_N(r) = - G M / r^2
    """
    r = np.asarray(r, dtype=float)
    return -G * M / r**2


def a_teu(r, M, lambda0, L1, L2, r0):
    """
    TEU effective acceleration using the rigidity profile.

    a_TEU(r) = a_N(r) * [ lambda(r0) / lambda(r) ]
    """
    r = np.asarray(r, dtype=float)
    lam_r0 = lambda_two_scale(r0, lambda0, L1, L2)
    lam_r = lambda_two_scale(r, lambda0, L1, L2)
    aN = a_newton(r, M)
    return aN * (lam_r0 / lam_r)


def rotation_curve_teu(M, lambda0, L1, L2, r0, r_array):
    """
    Toy rotation curves for a point–mass galaxy in TEU.

    Parameters
    ----------
    M : float
        Mass of the galaxy (kg).
    lambda0, L1, L2, r0 : floats
        TEU rigidity parameters.
    r_array : array_like
        Radii (m) at which to compute the circular velocity.

    Returns
    -------
    v_newton_kms : ndarray
        Newtonian circular velocity (km/s).
    v_teu_kms : ndarray
        TEU circular velocity (km/s).
    """
    r = np.asarray(r_array, dtype=float)
    aN = a_newton(r, M)
    aT = a_teu(r, M, lambda0, L1, L2, r0)

    vN = np.sqrt(r * np.abs(aN)) / 1000.0
    vT = np.sqrt(r * np.abs(aT)) / 1000.0

    return vN, vT

