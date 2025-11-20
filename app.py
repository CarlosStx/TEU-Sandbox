# app.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from teu_core import (
    G, Msun, a0,
    lambda_two_scale,
    a_newton,
    a_teu,
    rotation_curve_teu
)

# ---------------------------------------------------
# CONFIGURACIÓN DE LA APLICACIÓN
# ---------------------------------------------------
st.set_page_config(
    page_title="TEU Sandbox – Vacuum Rigidity",
    layout="wide"
)

st.title("TEU Sandbox – Structural Vacuum Rigidity Model")
st.markdown(
    """
    This interactive sandbox allows you to explore the **Theory of Empty Universe (TEU)** 
    vacuum rigidity model, based on a two–scale function λ(r).  
    Adjust the parameters using the controls on the left and observe how  
    the TEU acceleration, rigidity λ(r), and rotation curves behave.
    """
)

# ---------------------------------------------------
# SIDEBAR: CONTROLES
# ---------------------------------------------------
st.sidebar.header("Model Parameters")

# Mass of the system (galaxy)
M_log = st.sidebar.slider(
    "Log10(M / Msun)",
    min_value=9.0, max_value=12.0, value=11.0, step=0.1
)
M = (10**M_log) * Msun

lambda0 = st.sidebar.number_input(
    "λ₀ (reference rigidity)",
    min_value=0.1, max_value=10.0, value=1.0, step=0.1
)

# L1 and L2 in meters (log scale)
L1_log = st.sidebar.slider(
    "Log10(L₁ / m) [galactic scale]",
    min_value=18.0, max_value=22.0, value=20.5, step=0.1
)
L2_log = st.sidebar.slider(
    "Log10(L₂ / m) [cosmological scale]",
    min_value=24.0, max_value=28.0, value=26.0, step=0.1
)

L1 = 10**L1_log
L2 = 10**L2_log

# Reference radius r0
r0_log = st.sidebar.slider(
    "Log10(r₀ / m) [Newtonian reference]",
    min_value=18.0, max_value=21.0, value=19.0, step=0.1
)
r0 = 10**r0_log

# Radial range
r_min_log = st.sidebar.slider(
    "Log10(r_min / m)",
    min_value=17.0, max_value=21.0, value=18.0, step=0.1
)
r_max_log = st.sidebar.slider(
    "Log10(r_max / m)",
    min_value=21.0, max_value=28.0, value=26.0, step=0.1
)

r_min = 10**r_min_log
r_max = 10**r_max_log

N_pts = st.sidebar.slider(
    "Number of radial points",
    min_value=200, max_value=5000, value=1000, step=100
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Tips:**  
    - Move **L₁** to explore the MOND-like transition.  
    - Increase **L₂** to control the asymptotic (cosmological) acceleration.  
    - Modify **M** to simulate different galaxies.  
    """
)

# ---------------------------------------------------
# CÁLCULOS PRINCIPALES
# ---------------------------------------------------
r = np.logspace(np.log10(r_min), np.log10(r_max), N_pts)

aN = a_newton(r, M)
aT = a_teu(r, M, lambda0, L1, L2, r0)
lam = lambda_two_scale(r, lambda0, L1, L2)
epsilon = (aT - aN) / aN

# ---------------------------------------------------
# TABS VISUALES
# ---------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "Acceleration Profiles",
    "Vacuum Rigidity λ(r)",
    "Rotation Curves"
])

# ---------------------------------------------------
# TAB 1 — ACCELERATION
# ---------------------------------------------------
with tab1:
    st.subheader("Radial Acceleration: Newton vs TEU")

    fig, ax = plt.subplots(figsize=(7,5))
    ax.loglog(r, np.abs(aN), label="|a_N(r)| Newton")
    ax.loglog(r, np.abs(aT), "--", label="|a_TEU(r)| TEU")
    ax.set_xlabel("r (m)")
    ax.set_ylabel("|a(r)| (m/s²)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    st.markdown(
        """
        TEU matches Newtonian gravity at small radii,  
        deviates around the galactic scale **L₁**,  
        and approaches a nearly constant acceleration for **r >> L₂**.
        """
    )

    fig2, ax2 = plt.subplots(figsize=(7,5))
    ax2.semilogx(r, epsilon)
    ax2.set_xlabel("r (m)")
    ax2.set_ylabel("ε(r) = [a_TEU - a_N] / a_N")
    ax2.grid(True, which="both", alpha=0.3)
    st.pyplot(fig2)

# ---------------------------------------------------
# TAB 2 — LAMBDA PROFILE
# ---------------------------------------------------
with tab2:
    st.subheader("Structural Vacuum Rigidity λ(r)")

    fig3, ax3 = plt.subplots(figsize=(7,5))
    ax3.semilogx(r, lam)
    ax3.set_xlabel("r (m)")
    ax3.set_ylabel("λ(r)")
    ax3.grid(True, which="both", alpha=0.3)
    st.pyplot(fig3)

    st.markdown(
        f"""
        - For small radii, λ(r) ≈ λ₀ = **{lambda0:.2f}**  
        - Transition scale: **L₁ ≈ 10^{L1_log:.1f} m**  
        - Cosmological suppression scale: **L₂ ≈ 10^{L2_log:.1f} m**  
        """
    )

# ---------------------------------------------------
# TAB 3 — ROTATION CURVES
# ---------------------------------------------------
with tab3:
    st.subheader("Toy Rotation Curves: Newton vs TEU")

    kpc = 3.0857e19
    r_gal = np.linspace(1*kpc, 100*kpc, 500)
    r0_gal = 8 * kpc

    vN, vT = rotation_curve_teu(M, lambda0, L1, L2, r0_gal, r_gal)

    fig4, ax4 = plt.subplots(figsize=(7,5))
    ax4.plot(r_gal/kpc, vN, label="v_N(r) Newton")
    ax4.plot(r_gal/kpc, vT, "--", label="v_TEU(r) TEU")
    ax4.set_xlabel("r (kpc)")
    ax4.set_ylabel("v(r) (km/s)")
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    st.pyplot(fig4)

    st.markdown(
        """
        This toy model assumes a point-mass galaxy.  
        For realistic rotation curves, replace it with extended baryonic profiles  
        and compare directly with observational data (e.g., Gaia DR2).
        """
    )
