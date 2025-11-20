import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. CONSTANTES Y FUNCIONES TEU
# ==========================================

G = 6.674e-11           # Gravitational constant (SI)
Msun = 1.989e30         # Solar mass (kg)
a0 = 1.2e-10            # MOND scale (m/s^2), reference only


def lambda_two_scale(r, lambda0, L1, L2):
    """
    Two–scale structural vacuum rigidity:

        lambda(r) = lambda0 / (1 + r/L1 + (r/L2)^2)
    """
    r = np.asarray(r, dtype=float)
    return lambda0 / (1.0 + (r / L1) + (r / L2)**2)


def dlambda_dr(r, lambda0, L1, L2):
    """
    Radial derivative d lambda / dr for the two–scale model.

        lambda(r) = lambda0 / D
        D = 1 + r/L1 + (r/L2)^2

    d lambda/dr = -lambda0 * D' / D^2
                = -lambda0 * (1/L1 + 2r/L2^2) / (1 + r/L1 + (r/L2)^2)^2
    """
    r = np.asarray(r, dtype=float)
    D = 1.0 + (r / L1) + (r / L2)**2
    Dp = (1.0 / L1) + 2.0 * r / (L2**2)
    return -lambda0 * Dp / (D**2)


def a_newton(r, M):
    """
    Newtonian radial acceleration:

        a_N(r) = - G M / r^2
    """
    r = np.asarray(r, dtype=float)
    return -G * M / r**2


def a_teu(r, M, lambda0, L1, L2, r0):
    """
    TEU effective radial acceleration:

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
    Returns v_N(r) and v_TEU(r) in km/s.
    """
    r = np.asarray(r_array, dtype=float)
    aN = a_newton(r, M)
    aT = a_teu(r, M, lambda0, L1, L2, r0)

    vN = np.sqrt(r * np.abs(aN)) / 1000.0
    vT = np.sqrt(r * np.abs(aT)) / 1000.0

    return vN, vT


# ==========================================
# 2. CONFIGURACIÓN STREAMLIT
# ==========================================

st.set_page_config(
    page_title="TEU Sandbox – Structural Vacuum Rigidity",
    layout="wide"
)

st.title("TEU Sandbox – Structural Vacuum Rigidity Model")

st.markdown(
    r"""
This interactive sandbox explores the **Theory of the Empty Universe (TEU)**
using a two–scale structural vacuum rigidity function \(\lambda(r)\).

The model uses
\[
\lambda(r) = \frac{\lambda_0}{1 + r/L_1 + (r/L_2)^2},
\]
and an effective acceleration
\[
a_{\mathrm{TEU}}(r) = a_N(r)\,\frac{\lambda(r_0)}{\lambda(r)},
\qquad
a_N(r) = -\frac{GM}{r^2}.
\]
Use the sliders on the left to explore how the model behaves from
stellar to cosmological scales.
"""
)

# ==========================================
# 3. CONTROLES LATERALES
# ==========================================

st.sidebar.header("Model Parameters")

M_log = st.sidebar.slider(
    "Log10(M / Msun)",
    min_value=9.0, max_value=12.0, value=11.0, step=0.1
)
M = (10**M_log) * Msun

lambda0 = st.sidebar.number_input(
    "λ₀ (reference rigidity)",
    min_value=0.1, max_value=10.0, value=1.0, step=0.1
)

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

r0_log = st.sidebar.slider(
    "Log10(r₀ / m) [Newtonian reference]",
    min_value=18.0, max_value=21.0, value=19.0, step=0.1
)
r0 = 10**r0_log

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
**Tips**  
- Move **L₁** to explore the MOND–like transition.  
- Increase or decrease **L₂** to control the asymptotic acceleration.  
- Change **M** to mimic different galaxies.
"""
)

# ==========================================
# 4. CÁLCULOS PRINCIPALES
# ==========================================

r = np.logspace(np.log10(r_min), np.log10(r_max), N_pts)

aN = a_newton(r, M)
aT = a_teu(r, M, lambda0, L1, L2, r0)
lam = lambda_two_scale(r, lambda0, L1, L2)
epsilon = (aT - aN) / aN
dlam = dlambda_dr(r, lambda0, L1, L2)

# Regiones para estabilidad (umbrales arbitrarios pero útiles)
eps_GR = np.abs(epsilon) < 1e-3          # casi GR
eps_MOND = (np.abs(aT) / a0 > 0.3) & (np.abs(aT) / a0 < 3.0) & (r > 0.1*L1)
eps_div = lam < 1e-4                     # rigidez casi colapsada

# ==========================================
# 5. PESTAÑAS
# ==========================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Acceleration Profiles",
    "Vacuum Rigidity λ(r)",
    "Rotation Curves",
    "Field Equation Limits",
    "Stability Regions",
    "Comparison with Data"
])

# ---------- TAB 1: ACELERACIONES ----------
with tab1:
    st.subheader("Radial Acceleration: Newton vs TEU")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(r, np.abs(aN), label="|a_N(r)| Newton")
    ax.loglog(r, np.abs(aT), "--", label="|a_TEU(r)| TEU")
    ax.set_xlabel("r (m)")
    ax.set_ylabel("|a(r)| (m/s²)")
    ax.set_title("Radial acceleration: Newton vs TEU (log–log scale)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.semilogx(r, epsilon)
    ax2.set_xlabel("r (m)")
    ax2.set_ylabel("ε(r) = [a_TEU - a_N] / a_N")
    ax2.set_title("Relative deviation ε(r)")
    ax2.grid(True, which="both", alpha=0.3)
    st.pyplot(fig2)

# ---------- TAB 2: λ(r) ----------
with tab2:
    st.subheader("Structural Vacuum Rigidity λ(r)")

    fig3, ax3 = plt.subplots(figsize=(7, 5))
    ax3.semilogx(r, lam)
    ax3.axvline(L1, color="gray", linestyle=":", label="L₁")
    ax3.axvline(L2, color="black", linestyle="--", label="L₂")
    ax3.set_xlabel("r (m)")
    ax3.set_ylabel("λ(r)")
    ax3.set_title("Structural vacuum rigidity λ(r)")
    ax3.grid(True, which="both", alpha=0.3)
    ax3.legend()
    st.pyplot(fig3)

    st.markdown(
        f"""
- For **r → 0**, λ(r) → λ₀ ≈ **{lambda0:.2f}**.  
- At **r ≈ L₁ ≈ 10^{L1_log:.1f} m** the first transition (galactic) appears.  
- At **r ≈ L₂ ≈ 10^{L2_log:.1f} m** the cosmological scale controls the asymptotic behaviour.
        """
    )

# ---------- TAB 3: CURVAS DE ROTACIÓN ----------
with tab3:
    st.subheader("Toy Rotation Curves: Newton vs TEU")

    kpc = 3.0857e19
    r_gal = np.linspace(1 * kpc, 100 * kpc, 500)
    r0_gal = 8 * kpc

    vN, vT = rotation_curve_teu(M, lambda0, L1, L2, r0_gal, r_gal)

    # Datos aproximados Vía Láctea (Gaia/Eilers-like)
    r_obs_MW = np.array([5, 8, 10, 15, 20])   # kpc
    v_obs_MW = np.array([220, 232, 220, 210, 190])  # km/s

    fig4, ax4 = plt.subplots(figsize=(7, 5))
    ax4.plot(r_gal / kpc, vN, label="v_N(r) Newton")
    ax4.plot(r_gal / kpc, vT, "--", label="v_TEU(r) TEU")
    ax4.scatter(r_obs_MW, v_obs_MW, color="black",
                label="Milky Way (approx. data)")
    ax4.axvline(8, color='gray', linestyle=":", label="Sun (8 kpc)")
    ax4.set_xlabel("r (kpc)")
    ax4.set_ylabel("v(r) (km/s)")
    ax4.set_title("Milky Way–like rotation curves (point–mass toy model)")
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    st.pyplot(fig4)

    st.markdown(
        """
The black points are approximate Milky Way circular velocities from
Gaia/Eilers–type analyses.  
The TEU curve is generated using the same baryonic mass **M** and the
two–scale rigidity model.
"""
    )

# ---------- TAB 4: FIELD EQUATION LIMITS ----------
with tab4:
    st.subheader("Field Equation Limits and Asymptotic Behaviour")

    # Usamos algunos radios representativos
    r_samples = np.array([
        1e-3 * L1,   # r << L1
        0.1 * L1,
        L1,
        10 * L1,
        L2,
        10 * L2
    ])
    lam_s = lambda_two_scale(r_samples, lambda0, L1, L2)
    dlam_s = dlambda_dr(r_samples, lambda0, L1, L2)
    aN_s = a_newton(r_samples, M)
    aT_s = a_teu(r_samples, M, lambda0, L1, L2, r0)
    ratio_a = aT_s / aN_s

    st.markdown("**Sampled radii (relative to the TEU scales):**")
    for R, lR, dlR, rA in zip(r_samples, lam_s, dlam_s, ratio_a):
        st.markdown(
            f"- r = {R:.2e} m  "
            f"(r / L₁ = {R/L1:.2e}, r / L₂ = {R/L2:.2e})  —  "
            f"λ(r) = {lR:.3e},  dλ/dr = {dlR:.3e},  a_TEU/a_N = {rA:.3e}"
        )

    fig5, ax5 = plt.subplots(figsize=(7, 5))
    ax5.loglog(r, np.abs(dlam))
    ax5.axvline(L1, color="gray", linestyle=":", label="L₁")
    ax5.axvline(L2, color="black", linestyle="--", label="L₂")
    ax5.set_xlabel("r (m)")
    ax5.set_ylabel("|dλ/dr|")
    ax5.set_title("Derivative of the rigidity profile dλ/dr")
    ax5.grid(True, which="both", alpha=0.3)
    ax5.legend()
    st.pyplot(fig5)

    st.markdown(
        """
- For **r → 0**, \( \lambda(r) \to \lambda_0 \) and \( d\lambda/dr \to 0 \),
  so TEU coincides with Newton/GR.  
- Near **r ≈ L₁** the derivative is largest and the acceleration starts to
  deviate from the pure \(1/r^2\) law (MOND–like transition).  
- For **r ≫ L₂** the function saturates and produces an almost constant
  acceleration, mimicking a Hubble–like term.
"""
    )

# ---------- TAB 5: STABILITY REGIONS ----------
with tab5:
    st.subheader("Stability and Regime Classification")

    fig6, ax6 = plt.subplots(figsize=(7, 5))
    ax6.semilogx(r, epsilon, label="ε(r) = (a_TEU - a_N)/a_N")
    ax6.axhline(0.0, color="black", linewidth=0.7)

    # Pintamos franjas con fill_between
    ax6.fill_between(r, -1e-3, 1e-3, where=eps_GR,
                     color="green", alpha=0.2, label="GR-like (|ε| < 1e-3)")
    ax6.fill_between(r, np.min(epsilon), np.max(epsilon), where=eps_div,
                     color="red", alpha=0.1, label="Low-rigidity region")

    ax6.set_xlabel("r (m)")
    ax6.set_ylabel("ε(r)")
    ax6.set_title("Stability regions in TEU")
    ax6.grid(True, which="both", alpha=0.3)
    ax6.legend()
    st.pyplot(fig6)

    # Texto con rangos característicos
    r_GR = r[eps_GR]
    r_MOND = r[eps_MOND]
    r_div = r[eps_div]

    st.markdown("### Automatically detected regimes (approximate)")
    if r_GR.size > 0:
        st.markdown(
            f"- **GR-like region:** roughly from "
            f"{np.min(r_GR):.2e} m to {np.max(r_GR):.2e} m."
        )
    else:
        st.markdown("- **GR-like region:** none detected with |ε| < 1e-3.")

    if r_MOND.size > 0:
        st.markdown(
            f"- **MOND-like region (|a_TEU| ~ a₀):** roughly from "
            f"{np.min(r_MOND):.2e} m to {np.max(r_MOND):.2e} m."
        )
    else:
        st.markdown("- **MOND-like region:** not clearly present for this parameter set.")

    if r_div.size > 0:
        st.markdown(
            f"- **Low-rigidity / strong-deviation region:** starts around "
            f"{np.min(r_div):.2e} m and beyond."
        )
    else:
        st.markdown("- **Low-rigidity region:** λ(r) stays above 1e-4 over the plotted range.")

# ---------- TAB 6: COMPARISON WITH DATA ----------
with tab6:
    st.subheader("Comparison with Observational Data (toy samples)")

    kpc = 3.0857e19
    r_gal = np.linspace(1 * kpc, 100 * kpc, 500)
    r0_gal = 8 * kpc
    vN, vT = rotation_curve_teu(M, lambda0, L1, L2, r0_gal, r_gal)

    # Milky Way approximate
    r_MW = np.array([5, 8, 10, 15, 20])
    v_MW = np.array([220, 232, 220, 210, 190])

    # M31 approximate data (solo valores típicos)
    r_M31 = np.array([5, 10, 20, 30])
    v_M31 = np.array([250, 260, 250, 230])

    # SPARC-like toy galaxy
    r_SP = np.array([2, 4, 8, 12])
    v_SP = np.array([80, 110, 130, 140])

    fig7, ax7 = plt.subplots(figsize=(7, 5))
    ax7.plot(r_gal / kpc, vN, label="v_N(r) Newton")
    ax7.plot(r_gal / kpc, vT, "--", label="v_TEU(r) TEU")
    ax7.scatter(r_MW, v_MW, color="black", marker="o", label="Milky Way (approx.)")
    ax7.scatter(r_M31, v_M31, color="tab:orange", marker="s", label="M31 (approx.)")
    ax7.scatter(r_SP, v_SP, color="tab:green", marker="^", label="SPARC-like galaxy (toy)")
    ax7.set_xlabel("r (kpc)")
    ax7.set_ylabel("v(r) (km/s)")
    ax7.set_title("TEU rotation curves vs. approximate observational data")
    ax7.grid(True, alpha=0.3)
    ax7.legend()
    st.pyplot(fig7)

    st.markdown(
        """
The observational points are **illustrative toy data**:

- Milky Way: Gaia/Eilers–type circular velocity curve.  
- M31: typical high–mass spiral with \(v \sim 250\) km/s over 10–30 kpc.  
- “SPARC–like” galaxy: low–mass, slowly rising rotation curve.

The goal of this tab is **not** to provide a precise fit, but to show how
the TEU rotation curves can be visually compared with real galaxies and
used as a starting point for future detailed fitting.
"""
    )
