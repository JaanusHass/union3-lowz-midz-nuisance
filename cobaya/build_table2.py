#!/usr/bin/env python3
"""
Full-Planck Table 2 builder.
Reads all four chains (LCDM, LCDM+nuisance, CPL, CPL+nuisance), computes
best-fit chi2, Delta-chi2 vs LCDM, and AIC gain (= Dchi2 - 2k, paper convention).

Run where the chains live:
    python build_table2.py
"""

import numpy as np
from getdist import loadMCSamples
from scipy.special import erfinv

# ---- the four runs (roots) and their extra free params vs LCDM ----
MODELS = [
    # label,                root,                          k_extra (w0,wa,a_nuis)
    ("LCDM",                "chains/fullplanck_lcdm",       0),
    ("CPL",                 "chains/fullplanck_cpl",        2),
    ("LCDM + nuisance",     "chains/fullplanck_lcdm_nuis",  1),
    ("CPL + nuisance",      "chains/fullplanck",            3),
]
BURNIN = 0.0   # matches your margestats ("Removed no burn in"); set 0.3 to cross-check

def load(root):
    return loadMCSamples(root, settings={"ignore_rows": BURNIN})

def best_chi2(s):
    return float(np.min(s.getParams().chi2))

def mean_or_none(s, name):
    try:
        return float(s.mean(name))
    except Exception:
        return None

def sigma_from_lcdm(s):
    """2D distance of (w0,wa) posterior mean from LCDM."""
    mean = np.array([s.mean("w"), s.mean("wa")])
    cov  = np.array(s.cov(["w", "wa"]))
    d = mean - np.array([-1.0, 0.0])
    r = float(np.sqrt(d @ np.linalg.inv(cov) @ d))         # Mahalanobis
    P = 1 - np.exp(-r**2 / 2)                               # enclosed prob (2 dof)
    z = float(np.sqrt(2) * erfinv(P))                      # Gaussian-equivalent sigma
    return r, z

# ---- load everything ----
data = {}
for label, root, k in MODELS:
    s = load(root)
    data[label] = dict(
        s=s, k=k, chi2=best_chi2(s),
        H0=mean_or_none(s, "H0"),
        w0=mean_or_none(s, "w"),
        wa=mean_or_none(s, "wa"),
        anuis=mean_or_none(s, "a_nuis"),
    )

chi2_lcdm     = data["LCDM"]["chi2"]
chi2_lcdmnuis = data["LCDM + nuisance"]["chi2"]

def fmt(x, nd=3):
    return "   --  " if x is None else f"{x:+.{nd}f}"

# ---- main table ----
print("\n" + "=" * 92)
print("  FULL-PLANCK TABLE 2   (Planck NPIPE CamSpec TTTEEE + lowl TT/EE + DESI DR1 BAO + Union3)")
print("=" * 92)
hdr = f"  {'Model':16s} {'H0':>7s} {'w0':>8s} {'wa':>8s} {'a_nuis':>9s} {'bestchi2':>11s} {'Dchi2':>8s} {'AICgain':>8s}"
print(hdr)
print("  " + "-" * 88)
for label, root, k in MODELS:
    d = data[label]
    dchi2  = chi2_lcdm - d["chi2"]            # +ve = fits better than LCDM
    aicgain = dchi2 - 2 * d["k"]              # paper convention: +ve = favoured
    if label == "LCDM":
        dchi2_s, aic_s = "   --  ", "   --  "
    else:
        dchi2_s, aic_s = f"{dchi2:+.2f}", f"{aicgain:+.2f}"
    H0 = d["H0"]; H0s = f"{H0:.2f}" if H0 is not None else "  -- "
    print(f"  {label:16s} {H0s:>7s} {fmt(d['w0']):>8s} {fmt(d['wa']):>8s} "
          f"{fmt(d['anuis'],4):>9s} {d['chi2']:>11.3f} {dchi2_s:>8s} {aic_s:>8s}")
print("  " + "-" * 88)

# ---- KEY ROW: CPL+nuisance vs LCDM+nuisance ----
chi2_cplnuis = data["CPL + nuisance"]["chi2"]
dchi2_key  = chi2_lcdmnuis - chi2_cplnuis      # how much w0,wa improve on top of nuisance
aic_key    = dchi2_key - 2 * 2                 # 2 extra params (w0, wa)
print(f"\n  KEY COMPARISON  (does CPL add anything once nuisance is included?)")
print(f"    CPL+nuisance vs LCDM+nuisance :  Dchi2 = {dchi2_key:+.2f}   AIC gain = {aic_key:+.2f}"
      f"   -> {'CPL favoured' if aic_key > 0 else 'CPL DISFAVOURED'}")

# ---- 2D distance from LCDM for the CPL posteriors ----
print("\n  Dynamical-DE preference (2D distance of w0-wa mean from LCDM):")
for label in ["CPL", "CPL + nuisance"]:
    r, z = sigma_from_lcdm(data[label]["s"])
    print(f"    {label:16s}:  Mahalanobis r = {r:.2f}   (Gaussian-equiv {z:.2f} sigma)")

print("\n  NOTE: best chi2 = min over chain (MCMC proxy for global best fit).")
print("        For final publication numbers run  cobaya-run <input>.yaml --minimize  per model.")
print("=" * 92 + "\n")
