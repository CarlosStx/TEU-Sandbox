# TEU Sandbox – Structural Vacuum Rigidity Model  
Interactive exploration tool for the **Theory of the Empty Universe (TEU)**

![TEU Sandbox]([https://via.placeholder.com/1200x200?text=TEU+Sandbox](https://teu-sandbox-64zamcqlekuq5y7shq86zr.streamlit.app/#teu-sandbox-structural-vacuum-rigidity-model))

---

## 🔭 Overview

***TEU Sandbox** is an interactive tool to explore the gravitational behavior 
predicted by TEU.  

The central element of the model is the *structural vacuum rigidity*:

$$
\lambda(r) = \frac{\lambda_0}{1 + r/L_1 + (r/L_2)^2}
$$

and the effective TEU radial acceleration:

$$
a_{\mathrm{TEU}}(r) = a_{N}(r)\,\frac{\lambda(r_0)}{\lambda(r)}.
$$

where Newtonian gravity is:

$$
a_{N}(r) = -\frac{GM}{r^2}.
$$

The two characteristic scales of TEU are:

- **L₁** – Galactic transition scale (MOND-like regime)  
- **L₂** – Cosmological scale (Hubble-like acceleration)  


This repository provides:

- A fully interactive **Streamlit** interface  
- Numerical evaluation of TEU accelerations  
- Vacuum rigidity profiles  
- Comparison vs Newtonian gravity  
- Rotation curves (Newton vs TEU)  
- Stability and asymptotic regime diagnostics  
- Approximate observational data overlays (Milky Way, M31, SPARC-like)

The goal is to offer physicists and astronomers a **sandbox** to test,
probe, and extend the TEU framework across galactic and cosmological scales.

---

## 📂 Repository Structure


---

## 🚀 Running the Sandbox Locally

Clone the repository:

```bash
git clone https://github.com/<your-user>/TEU-Sandbox.git
cd TEU-Sandbox

pip install -r requirements.txt

http://localhost:8501


