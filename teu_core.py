import numpy as np

G    = 6.674e-11
Msun = 1.989e30
a0   = 1.2e-10

def lambda_two_scale(r, lambda0, L1, L2):
    return lambda0 / (1.0 + r/L1 + (r/L2)**2)

def a_newton(r, M):
    return -G * M / r**2

def a_teu(r, M, lambda0, L1, L2, r0):
    lam_r0 = lambda_two_scale(r0, lambda0, L1, L2)
    lam_r  = lambda_two_scale(r,  lambda0, L1, L2)
    return a_newton(r, M) * (lam_r0 / lam_r)

def rotation_curve_teu(M, lambda0, L1, L2, r0, r_array):
    aN = a_newton(r_array, M)
    aT = a_teu(r_array, M, lambda0, L1, L2, r0)
    vN = np.sqrt(r_array * np.abs(aN)) / 1000.0
    vT = np.sqrt(r_array * np.abs(aT)) / 1000.0
    return vN, vT
