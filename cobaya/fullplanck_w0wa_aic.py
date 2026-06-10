#!/usr/bin/env python3
"""
Full-Planck (NPIPE CamSpec TTTEEE + lowl TT/EE + DESI DR1 BAO + Union3)
CPL vs CPL+nuisance: exact best-fit chi2 / AIC + publication w0-wa figure.

Run on the WSL2 machine where the chains live (getdist ships with Cobaya):
    python3 fullplanck_w0wa_aic.py

Outputs:
    fullplanck_w0wa.pdf  / .png   -- two-contour w0-wa figure (+ LCDM star)
    prints best-fit chi2, Delta chi2, Delta AIC, and 2D distance from LCDM
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from getdist import plots, loadMCSamples

# ---------------------------------------------------------------- config
ROOT_CPL  = "chains/fullplanck_cpl"   # CPL, no nuisance
ROOT_NUIS = "chains/fullplanck"       # CPL + Union3 nuisance (a_nuis)

# Your margestats said "Removed no burn in". Keep 0 to match those numbers,
# or set 0.3 if you want a conservative burn-in cut.
IGNORE_ROWS = 0

# Extra free parameters of each model vs LCDM (w0,wa fixed):
#   CPL          : w0, wa            -> 2
#   CPL+nuisance : w0, wa, a_nuis    -> 3
# For the CPL+nuisance vs CPL comparison only the difference matters: dk = 1.
DK_NUIS_MINUS_CPL = 1

COL_CPL  = "#2c7fb8"   # blue  = CPL (no nuisance)
COL_NUIS = "#41ab5d"   # green = CPL + nuisance
# ------------------------------------------------------------------------

s_cpl  = loadMCSamples(ROOT_CPL,  settings={"ignore_rows": IGNORE_ROWS})
s_nuis = loadMCSamples(ROOT_NUIS, settings={"ignore_rows": IGNORE_ROWS})

# Parameter names in your chains: 'w' (=w0, label w_0) and 'wa'.
PW, PWA = "w", "wa"


def best_chi2(s):
    """Minimum chi2 over the chain = MCMC proxy for the best fit."""
    return float(np.min(s.getParams().chi2))


def sigma_from_lcdm(s):
    """2D Mahalanobis distance of the (w0,wa) posterior mean from LCDM."""
    mean = np.array([s.mean(PW), s.mean(PWA)])
    cov = np.array(s.cov([PW, PWA]))
    d = mean - np.array([-1.0, 0.0])
    return float(np.sqrt(d @ np.linalg.inv(cov) @ d))


chi2_cpl, chi2_nuis = best_chi2(s_cpl), best_chi2(s_nuis)
dchi2 = chi2_nuis - chi2_cpl
dAIC = dchi2 + 2 * DK_NUIS_MINUS_CPL

print("\n================  CPL  vs  CPL+nuisance  (full Planck)  ================")
print(f"  best-fit chi2  CPL           = {chi2_cpl:.3f}")
print(f"  best-fit chi2  CPL+nuisance  = {chi2_nuis:.3f}")
print(f"  Delta chi2_min (nuis - cpl)  = {dchi2:+.3f}")
print(f"  Delta AIC      (dk = {DK_NUIS_MINUS_CPL})       = {dAIC:+.3f}   "
      f"({'CPL+nuisance favoured' if dAIC < 0 else 'CPL favoured'})")
print("  ----------------------------------------------------------------------")
print(f"  w0, wa  CPL           : {s_cpl.mean(PW):+.3f}, {s_cpl.mean(PWA):+.3f}"
      f"   -> {sigma_from_lcdm(s_cpl):.2f}sigma from LCDM (2D)")
print(f"  w0, wa  CPL+nuisance  : {s_nuis.mean(PW):+.3f}, {s_nuis.mean(PWA):+.3f}"
      f"   -> {sigma_from_lcdm(s_nuis):.2f}sigma from LCDM (2D)")
print("  NOTE: min(chi2) over a chain approximates the global best fit; for a")
print("  publication number run  cobaya-run <input>.yaml --minimize  per model.")
print("========================================================================\n")

# ----------------------------------------------------------------- figure
g = plots.get_single_plotter(width_inch=5.2, ratio=0.85)
g.settings.axes_fontsize = 12
g.settings.axes_labelsize = 16
g.settings.alpha_filled_add = 0.75
g.settings.figure_legend_frame = False

g.plot_2d([s_cpl, s_nuis], PW, PWA, filled=True, colors=[COL_CPL, COL_NUIS])

ax = g.subplots[0, 0]
ax.plot(-1, 0, marker="*", ms=17, color="red", mec="k", mew=0.6, zorder=20)

# keep the LCDM star comfortably inside the frame
x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
ax.set_xlim(min(x0, -1.10), x1)
ax.set_ylim(min(y0, -0.15), max(y1, 0.20))

handles = [
    Patch(facecolor=COL_CPL,  label="CPL (no nuisance)"),
    Patch(facecolor=COL_NUIS, label="CPL + nuisance"),
    Line2D([0], [0], marker="*", color="none", markerfacecolor="red",
           markeredgecolor="k", markersize=13,
           label=r"$\Lambda$CDM ($w_0{=}{-}1,\ w_a{=}0$)"),
]
ax.legend(handles=handles, loc="upper right", fontsize=10,
          framealpha=0.9, handlelength=1.4)

ax.set_xlabel(r"$w_0$")
ax.set_ylabel(r"$w_a$")

for ext in ("pdf", "png"):
    g.export(f"fullplanck_w0wa.{ext}")
print("wrote fullplanck_w0wa.pdf and fullplanck_w0wa.png")
