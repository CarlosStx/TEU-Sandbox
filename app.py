import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. CONSTANTES Y FUNCIONES TEU (NUEVAS)
# ==========================================

G = 6.674e-11           # Constante de gravitación (SI)
Msun = 1.989e30         # Masa solar (kg)
a0 = 1.2e-10            # Escala MOND (m/s^2), solo referencia


def lambda_two_scale(r, lambda0, L1, L2):
    """
    Rigidez estructural del vacío con dos escalas:

        lambda(r) = lambda0 / (1 + r/L1 + (r/L2)^2)
    """
    r = np.asarray(r, dtype=float)
    return lambda0 / (1.0 + (r / L1) + (r / L2)**2)


def a_newton(r, M):
    """
    Aceleración radial newtoniana:

        a_N(r) = - G M / r^2
    """
    r = np.asarray(r, dtype=float)
    return -G * M / r**2


def a_teu(r, M, lambda0, L1, L2, r0):
    """
    Aceleración efectiva TEU:

        a_TEU(r) = a_N(r) * [ lambda(r0) / lambda(r) ]
    """
    r = np.asarray(r, dtype=float)
    lam_r0 = lambda_two_scale(r0, lambda0, L1, L2)
    lam_r = lambda_two_scale(r, lambda0, L1, L2)
    aN = a_newton(r, M)
    return aN * (lam_r0 / lam_r)


def rotation_curve_teu(M, lambda0, L1, L2, r0, r_array):
    """
    Curvas de rotación toy para una galaxia tipo punto-masa en TEU.
    Devuelve v_N(r) y v_TEU(r) en km/s.
    """
    r = np.asarray(r_array, dtype=float)
    aN = a_newton(r, M)
    aT = a_teu(r, M, lambda0, L1, L2, r0)

    vN = np.sqrt(r * np.abs(aN)) / 1000.0
    vT = np.sqrt(r * np.abs(aT)) / 1000.0

    return vN, vT


# ==========================================
# 2. CONFIGURACIÓN DE LA APP STREAMLIT
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

The model uses:
\[
\lambda(r) = \frac{\lambda_0}{1 + r/L_1 + (r/L_2)^2}
\]
and an effective acceleration
\[
a_{\mathrm{TEU}}(r) = a_N(r)\,\frac{\lambda(r_0)}{\lambda(r)},
\qquad
a_N(r) = -\frac{GM}{r^2}.
\]
"""
)

# ==========================================
# 3. CONTROLES EN LA BARRA LATERAL
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
- Increase **L₂** to control the asymptotic acceleration.  
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

# ==========================================
# 5. TABS PRINCIPALES
# ==========================================

tab1, tab2, tab3 = st.tabs([
    "Acceleration Profiles",
    "Vacuum Rigidity λ(r)",
    "Rotation Curves"
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
    ax3.set_xlabel("r (m)")
    ax3.set_ylabel("λ(r)")
    ax3.set_title("Structural vacuum rigidity λ(r)")
    ax3.grid(True, which="both", alpha=0.3)
    st.pyplot(fig3)

    st.markdown(
        f"""
- For small radii, λ(r) ≈ λ₀ = **{lambda0:.2f}**  
- Galactic scale L₁ ≈ 10^{L1_log:.1f} m  
- Cosmological scale L₂ ≈ 10^{L2_log:.1f} m  
        """
    )

# ---------- TAB 3: CURVAS DE ROTACIÓN ----------
with tab3:
    st.subheader("Toy Rotation Curves: Newton vs TEU")

    kpc = 3.0857e19
    r_gal = np.linspace(1 * kpc, 100 * kpc, 500)
    r0_gal = 8 * kpc

    vN, vT = rotation_curve_teu(M, lambda0, L1, L2, r0_gal, r_gal)

    fig4, ax4 = plt.subplots(figsize=(7, 5))
    ax4.plot(r_gal / kpc, vN, label="v_N(r) Newton")
    ax4.plot(r_gal / kpc, vT, "--", label="v_TEU(r) TEU")
    ax4.set_xlabel("r (kpc)")
    ax4.set_ylabel("v(r) (km/s)")
    ax4.set_title("Milky Way–like rotation curves (point–mass toy model)")
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    st.pyplot(fig4)

    st.markdown(
        """
This is a **toy model** assuming a point–mass galaxy.  
For realistic fits, one can plug in extended baryonic profiles 
(disk + bulge) and compare directly with Gaia/Eilers-type data.
"""
    )
