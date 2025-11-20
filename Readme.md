# TEU Sandbox – Structural Vacuum Rigidity Model

This repository contains an interactive **Streamlit** application to explore the
two–scale structural vacuum rigidity model proposed in the **Theory of the Empty Universe (TEU)**.

The effective rigidity of the vacuum is modeled as

\[
\lambda(r) = \frac{\lambda_0}{1 + r/L_1 + (r/L_2)^2},
\]

and the TEU radial acceleration is defined by

\[
a_{\text{TEU}}(r) = a_N(r)\,\frac{\lambda(r_0)}{\lambda(r)},
\qquad
a_N(r) = -\frac{GM}{r^2}.
\]

The sandbox allows you to:

- Vary the **galaxy mass** \(M\),
- Tune the **galactic scale** \(L_1\) (MOND-like transition),
- Tune the **cosmological scale** \(L_2\) (asymptotic Hubble-like acceleration),
- Choose a **reference radius** \(r_0\),
- Inspect:
  - Newton vs TEU accelerations,
  - the rigidity profile \(\lambda(r)\),
  - toy rotation curves for a Milky Way–like galaxy.

## How to run locally

```bash
git clone https://github.com/<your-user>/TEU-Sandbox.git
cd TEU-Sandbox
pip install -r requirements.txt
streamlit run app.py
