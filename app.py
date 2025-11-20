import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ================================================================
# TEU SANDBOX – app.py
# Structural vacuum rigidity with two scales (L1, L2)
# ================================================================

st.set_page_config(
    page_title="TEU Sandbox",
    layout="wide",
)

st.title("TEU Sandbox – Structural Vacuum Rigidity")
st.markdown(
    """
Interactive sandbox for the **Theory of the Empty Universe (TEU)**.

The model assumes that gravity emerges from a deformation of the vacuum,
encoded in a scale–dependent rigidity function
\\(\\lambda(r)\\). The effective radial acceleration is

\\[
a_{\\rm TEU}(r) = a_{\\rm N}(r)\\,\\frac{\\lambda(r_0)}{\\lambda(r)},
\\quad
a_{\\rm N}(r) = -\\,\\frac{GM}{r^2},
\\]

with a two–scale rigidity

\\[
\\lambda(r) = \\frac{\\lambda_0}
{1 + r/L_1 + (r/L_2)^2 }.
\\]

Use the controls in the sidebar to explore Newtonian, MOND–like and
cosmological regimes.
"""
)

# ================================================================
# CONSTANTS
# ================================================================

G = 6.674e-11          # gravitational constant (SI)
Msun = 1.989e30        # solar mass (kg)
a0 = 1.2e-10           # MOND scale (m/s^2), for reference

# ================================================================
# SIDEBAR – MODEL PARAMETERS
# ================================================================

st.sidebar.title("TEU Sandbox – Model Parameters")
st.sidebar.markdown(
    "Adjust the **physical scales** of the TEU model and the "
    "**numerical domain** used in the plots."
)

# ------------------ 1. Physical parameters -----------------------
with st.sidebar.expander("1️⃣ Physical parameters", expanded=True):

    log10_M = st.slider(
        "log₁₀(M / M☉)",
        min_value=9.0,
        max_value=12.5,
        value=11.0,
        step=0.1,
        help="Total baryonic mass of the system in solar masses. "
             "Typical galaxies: 10¹⁰–10¹² M☉."
    )

    lambda0 = st.number_input(
        "λ₀ (reference rigidity)",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Vacuum rigidity in the inner Newtonian/GR regime. "
             "λ₀ = 1 is the natural normalization."
    )

    log10_L1 = st.slider(
        "log₁₀(L₁ / m) – galactic scale",
        min_value=19.0,
        max_value=21.5,
        value=20.5,
        step=0.1,
        help="Transition scale to the MOND-like regime. "
             "Of order 10²⁰ m (~a few–tens of kpc)."
    )

    log10_L2 = st.slider(
        "log₁₀(L₂ / m) – cosmological scale",
        min_value=25.0,
        max_value=27.5,
        value=26.0,
        step=0.1,
        help="Transition scale to the cosmological regime, where "
             "a_TEU tends to an almost constant value (Hubble-like)."
    )

# ------------------ 2. Reference & radial domain -----------------
with st.sidebar.expander("2️⃣ Reference & radial domain", expanded=True):

    log10_r0 = st.slider(
        "log₁₀(r₀ / m) – Newtonian reference",
        min_value=18.0,
        max_value=21.0,
        value=19.0,
        step=0.1,
        help="Radius where we impose a_TEU(r₀) = a_N(r₀). "
             "Typically inside the Newtonian/GR regime."
    )

    log10_r_min = st.slider(
        "log₁₀(r_min / m)",
        min_value=17.0,
        max_value=21.0,
        value=18.0,
        step=0.1,
        help="Inner radius used in all radial plots. "
             "Should be smaller than r₀."
    )

    log10_r_max = st.slider(
        "log₁₀(r_max / m)",
        min_value=23.0,
        max_value=27.0,
        value=26.0,
        step=0.1,
        help="Outer radius used in all radial plots. "
             "Must be large enough to reach the L₂ / cosmological regime."
    )

# ------------------ 3. Numerical settings ------------------------
with st.sidebar.expander("3️⃣ Numerical settings", expanded=True):

    N_pts = st.slider(
        "Number of radial points",
        min_value=300,
        max_value=4000,
        value=1000,
        step=100,
        help="Resolution of the radial grid. "
             "Higher values give smoother curves but require more CPU time."
    )

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Tip:** Start with the default values to see the three regimes "
    "(Newton/GR, MOND-like, cosmological) and then adjust L₁ and L₂ "
    "to explore how the TEU rigidity changes gravity."
)

# --------- Convert log-parameters to physical values --------------
M     = (10.0 ** log10_M) * Msun
L1    = 10.0 ** log10_L1
L2    = 10.0 ** log10_L2
r0    = 10.0 ** log10_r0
r_min = 10.0 ** log10_r_min
r_max = 10.0 ** log10_r_max

# radial grid
r = np.logspace(np.log10(r_min), np.log10(r_max), N_pts)

# ================================================================
# PHYSICAL FUNCTIONS
# ================================================================

def lambda_r(r_val: np.ndarray) -> np.ndarray:
    """Two–scale structural vacuum rigidity λ(r)."""
    return lambda0 / (1.0 + (r_val / L1) + (r_val / L2) ** 2)


def a_newton(r_val: np.ndarray) -> np.ndarray:
    """Newtonian radial acceleration for a point mass M."""
    return -G * M / r_val**2


def a_teu(r_val: np.ndarray) -> np.ndarray:
    """
    TEU effective acceleration:
    a_TEU(r) = a_N(r) * [ lambda(r0) / lambda(r) ].
    """
    lam0 = lambda_r(r0)
    lam = lambda_r(r_val)
    return a_newton(r_val) * (lam0 / lam)


# precompute main arrays
lam_arr = lambda_r(r)
aN_arr = a_newton(r)
aT_arr = a_teu(r)
epsilon = (aT_arr - aN_arr) / aN_arr

# ================================================================
# TABS
# ================================================================

(
    tab_acc,
    tab_lambda,
    tab_rot,
    tab_limits,
    tab_stab,
    tab_comp,
) = st.tabs(
    [
        "Acceleration Profiles",
        "Vacuum Rigidity λ(r)",
        "Rotation Curves",
        "Field Equation Limits",
        "Stability Regions",
        "Comparison with Data",
    ]
)

# ------------------------------------------------
# TAB 1 – Acceleration Profiles
# ------------------------------------------------
with tab_acc:
    st.subheader("Radial Acceleration: Newton vs TEU")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.loglog(r, np.abs(aN_arr), label="|a_N(r)| Newton")
    ax.loglog(r, np.abs(aT_arr), "--", label="|a_TEU(r)| TEU")
    ax.set_xlabel("r (m)")
    ax.set_ylabel("|a(r)| (m/s²)")
    ax.set_title("Radial acceleration (log–log scale)")
    ax.grid(True, which="both", ls=":")
    ax.legend()
    st.pyplot(fig)

    st.markdown("### Relative deviation ε(r)")
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.semilogx(r, epsilon)
    ax2.axhline(0, color="gray", lw=0.8)
    ax2.set_xlabel("r (m)")
    ax2.set_ylabel("ε(r) = [a_TEU − a_N]/a_N")
    ax2.grid(True, which="both", ls=":")
    st.pyplot(fig2)

# ------------------------------------------------
# TAB 2 – Vacuum Rigidity λ(r)
# ------------------------------------------------
with tab_lambda:
    st.subheader("Structural Vacuum Rigidity λ(r)")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(r, lam_arr)
    ax.set_xlabel("r (m)")
    ax.set_ylabel("λ(r)")
    ax.set_title("Vacuum rigidity profile")
    ax.grid(True, which="both", ls=":")
    st.pyplot(fig)

    # derivative dλ/dr
    dlam_dr = np.gradient(lam_arr, r)
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.semilogx(r, dlam_dr)
    ax2.set_xlabel("r (m)")
    ax2.set_ylabel("dλ/dr")
    ax2.set_title("Derivative of the rigidity profile")
    ax2.grid(True, which="both", ls=":")
    st.pyplot(fig2)

# ------------------------------------------------
# TAB 3 – Rotation Curves (spherical)
# ------------------------------------------------
with tab_rot:
    st.subheader("Spherical Rotation Curves")

    kpc = 3.0857e19
    r_kpc = r / kpc

    vN = np.sqrt(r * np.abs(aN_arr)) / 1000.0  # km/s
    vT = np.sqrt(r * np.abs(aT_arr)) / 1000.0  # km/s

    # approximate Milky Way data (Gaia/Eilers-like)
    r_obs_kpc = np.array([5, 8, 10, 15, 20])
    v_obs_kms = np.array([220, 232, 220, 210, 190])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(r_kpc, vN, label="v_N(r) Newton")
    ax.plot(r_kpc, vT, "--", label="v_TEU(r) TEU")
    ax.scatter(r_obs_kpc, v_obs_kms, c="k", label="Milky Way (approx.)")
    ax.axvline(8, color="gray", linestyle=":", label="Sun (8 kpc)")
    ax.set_xlim(0.5, 100)
    ax.set_xlabel("r (kpc)")
    ax.set_ylabel("v(r) (km/s)")
    ax.set_title("Milky Way–type rotation curve (spherical mass)")
    ax.grid(True, ls=":")
    ax.legend()
    st.pyplot(fig)

    st.markdown(
        """
The TEU curve can remain nearly flat over 5–20 kpc while the Newtonian
curve (with the same baryonic mass) decreases too quickly, failing to
reproduce the observed rotation curve.
"""
    )

# ------------------------------------------------
# TAB 4 – Field Equation Limits
# ------------------------------------------------
with tab_limits:
    st.subheader("Field Equation Limits and Asymptotic Regimes")

    st.markdown(
        r"""
For the two–scale rigidity
\[
\lambda(r) = \frac{\lambda_0}{1 + r/L_1 + (r/L_2)^2},
\]
we can identify four qualitative regimes:

- **Inner Newtonian/GR**: \( r \ll L_1 \Rightarrow \lambda(r)\approx \lambda_0 \).
- **MOND-like transition**: \( r\sim L_1 \Rightarrow \lambda(r) \) starts to drop and
  TEU deviates from the pure \(1/r^2\) law.
- **Intermediate regime**: \( L_1 \ll r \ll L_2 \Rightarrow \lambda(r)\sim L_1/r \).
- **Cosmological regime**: \( r \gtrsim L_2 \Rightarrow \lambda(r)\sim L_2^2/r^2 \) and
  \( a_{\rm TEU}(r) \) tends to an almost constant value.
"""
    )

    # pick representative radii
    r_inner = L1 * 1e-2
    r_L1 = L1
    r_mid = np.sqrt(L1 * L2)
    r_L2 = L2
    r_far = L2 * 10

    sample_r = np.array([r_inner, r_L1, r_mid, r_L2, r_far])
    labels = ["r ≪ L₁", "r = L₁", "√(L₁L₂)", "r = L₂", "10 L₂"]

    rows = []
    for lbl, R in zip(labels, sample_r):
        lam = lambda_r(R)
        aN = a_newton(R)
        aT = a_teu(R)
        eps = (aT - aN) / aN
        rows.append(
            {
                "Regime": lbl,
                "r (m)": f"{R:.3e}",
                "λ(r)": f"{lam:.3e}",
                "a_N (m/s²)": f"{aN:.3e}",
                "a_TEU (m/s²)": f"{aT:.3e}",
                "ε = (a_TEU−a_N)/a_N": f"{eps:.3e}",
            }
        )

    st.table(rows)

# ------------------------------------------------
# TAB 5 – Stability Regions
# ------------------------------------------------
with tab_stab:
    st.subheader("Stability / Agreement Regions")

    abs_eps = np.abs(epsilon)

    # thresholds
    thr_GR = 1e-3      # GR-like
    thr_MOND = 0.1     # mild deviation

    def range_from_mask(mask):
        if not np.any(mask):
            return "—"
        r_sel = r[mask]
        return f"[{r_sel.min():.3e}, {r_sel.max():.3e}] m"

    mask_GR = abs_eps < thr_GR
    mask_MOND = (abs_eps >= thr_GR) & (abs_eps < thr_MOND)
    mask_strong = abs_eps >= thr_MOND

    st.markdown("**Radial ranges (in meters):**")

    st.write(
        f"- **GR-like regime** (|ε| < 10⁻³): "
        f"{range_from_mask(mask_GR)}"
    )
    st.write(
        f"- **MOND-like / mild deviation** (10⁻³ ≤ |ε| < 0.1): "
        f"{range_from_mask(mask_MOND)}"
    )
    st.write(
        f"- **Strong deviation** (|ε| ≥ 0.1): "
        f"{range_from_mask(mask_strong)}"
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(r, abs_eps)
    ax.axhline(thr_GR, color="green", linestyle="--", label="GR limit (10⁻³)")
    ax.axhline(thr_MOND, color="orange", linestyle="--", label="MOND-like (0.1)")
    ax.set_xlabel("r (m)")
    ax.set_ylabel("|ε(r)|")
    ax.set_title("Absolute deviation |ε(r)|")
    ax.grid(True, which="both", ls=":")
    ax.legend()
    st.pyplot(fig)

# ------------------------------------------------
# TAB 6 – Comparison with Data (MW + M31)
# ------------------------------------------------
with tab_comp:
    st.subheader("Comparison with Observational Data")

    kpc = 3.0857e19
    r_kpc = r / kpc
    vN = np.sqrt(r * np.abs(aN_arr)) / 1000.0
    vT = np.sqrt(r * np.abs(aT_arr)) / 1000.0

    # Milky Way data (approx.)
    r_MW = np.array([5, 8, 10, 15, 20])
    v_MW = np.array([220, 232, 220, 210, 190])

    # Very rough M31-like data (illustrative only)
    r_M31 = np.array([5, 10, 15, 20, 25])
    v_M31 = np.array([250, 260, 250, 240, 230])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(r_kpc, vN, label="v_N(r) Newton (spherical)")
    ax.plot(r_kpc, vT, "--", label="v_TEU(r) TEU (spherical)")
    ax.scatter(r_MW, v_MW, c="k", marker="o", label="Milky Way (approx.)")
    ax.scatter(r_M31, v_M31, c="red", marker="x", label="M31 (illustrative)")
    ax.set_xlim(0.5, 50)
    ax.set_xlabel("r (kpc)")
    ax.set_ylabel("v(r) (km/s)")
    ax.set_title("Comparison with approximate rotation–curve data")
    ax.grid(True, ls=":")
    ax.legend()
    st.pyplot(fig)

    st.markdown(
        """
**Notes**

- The observational points are approximate and only meant to illustrate
  the qualitative agreement of TEU with flat rotation curves using
  **baryonic mass only**.
- For a precise comparison one should use full SPARC data and a
  realistic baryonic mass model (disk + bulge + gas) instead of a
  single spherical mass.
"""
    )
