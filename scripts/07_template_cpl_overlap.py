#!/usr/bin/env python
"""
07_template_cpl_overlap.py

Template-CPL overlap (degeneracy) test and low-z / mid-z component decomposition for the
Union3 compressed-data nuisance diagnostic (Sections 7.5 and 7.6 of the note).

Addresses the concern that the template merely reproduces the CPL distance deformation:
- the angle between T and the CPL direction in the precision metric (M marginalized);
- the nested Delta chi2 ledger (CPL-after-T vs T-after-CPL);
- the recovered amplitude with and without CPL in the model;
- decomposition of T into its low-z and mid-z Gaussian arms.

Outputs:
    results/template_cpl_overlap.csv
    results/component_decomposition.csv
    paper/overlap_angle.pdf, paper/overlap_angle.png   (Figure 2 of the note)

Expected (fixed-Omega_m = 0.2975 protocol):
    cos theta ~ 0.79 (theta ~ 38 deg); 63% of T inside the CPL subspace
    CPL after T ~ 0.87 ; T after CPL ~ 10.45
    amplitude a: 0.0279 (LCDM+nuisance) -> 0.0357 (CPL+nuisance)

Required input file:
    data/mu_mat_union3_cosmo=2_mu.fits

Run from repository root:
    python scripts/07_template_cpl_overlap.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize

try:
    from astropy.io import fits
except ImportError as exc:
    raise SystemExit("Missing dependency: astropy. Install with: pip install astropy") from exc


C_KM_S = 299792.458
OMEGA_M_FIXED = 0.2975
H0_FIXED = 70.0


def load_union3_compressed(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Download mu_mat_union3_cosmo=2_mu.fits "
            "from rubind/union3_release and place it in data/."
        )
    with fits.open(path) as hdul:
        mat = np.asarray(hdul[0].data, dtype=float)
    z = np.asarray(mat[0, 1:], dtype=float)
    mu = np.asarray(mat[1:, 0], dtype=float)
    precision = np.asarray(mat[1:, 1:], dtype=float)
    order = np.argsort(z)
    return z[order], mu[order], precision[np.ix_(order, order)]


def e_z(z, omega_m, w0=-1.0, wa=0.0):
    z = np.asarray(z, dtype=float)
    de = (1.0 + z) ** (3.0 * (1.0 + w0 + wa)) * np.exp(-3.0 * wa * z / (1.0 + z))
    ez2 = omega_m * (1.0 + z) ** 3 + (1.0 - omega_m) * de
    if np.any(ez2 <= 0.0) or not np.all(np.isfinite(ez2)):
        return np.full_like(z, np.nan)
    return np.sqrt(ez2)


def distance_modulus(z, omega_m=OMEGA_M_FIXED, h0=H0_FIXED, w0=-1.0, wa=0.0):
    z = np.asarray(z, dtype=float)
    grid = np.linspace(0.0, float(np.max(z)) * 1.001 + 1e-6, 6000)
    ez = e_z(grid, omega_m=omega_m, w0=w0, wa=wa)
    if np.any(~np.isfinite(ez)):
        return np.full_like(z, np.nan)
    integral = cumulative_trapezoid(1.0 / ez, grid, initial=0.0)
    chi = np.interp(z, grid, integral)
    dl_mpc = (C_KM_S / h0) * (1.0 + z) * chi
    if np.any(dl_mpc <= 0.0):
        return np.full_like(z, np.nan)
    return 5.0 * np.log10(dl_mpc) + 25.0


def gaussian(z, c, s):
    return np.exp(-0.5 * ((z - c) / s) ** 2)


def standardize(x):
    y = np.asarray(x, dtype=float) - np.mean(x)
    return y / np.sqrt(np.mean(y ** 2))


def gls(mu_obs, precision, mu_model, template=None):
    """Return (chi2, beta) with M (and optional template amplitude) marginalized."""
    residual = mu_obs - mu_model
    xmat = np.ones((len(mu_obs), 1)) if template is None \
        else np.column_stack([np.ones(len(mu_obs)), template])
    fisher = xmat.T @ precision @ xmat
    beta = np.linalg.pinv(fisher) @ (xmat.T @ precision @ residual)
    final = residual - xmat @ beta
    return float(final.T @ precision @ final), beta


def fit_cpl(z, mu_obs, precision, template=None):
    """Return (chi2, w0, wa) at fixed Omega_m; Nelder-Mead seed then bounded L-BFGS-B."""
    def obj(x):
        return gls(mu_obs, precision,
                   distance_modulus(z, OMEGA_M_FIXED, H0_FIXED, float(x[0]), float(x[1])),
                   template)[0]
    bounds = [(-3.0, 1.0), (-5.0, 5.0)]
    starts = [(-1.0, 0.0), (-0.7, -1.0), (-1.2, 1.0), (-0.5, -2.0), (-1.5, 2.0)]
    best = None
    for s in starts:
        nm = minimize(obj, np.array(s), method="Nelder-Mead", options={"maxiter": 5000})
        x0 = np.array([np.clip(nm.x[0], *bounds[0]), np.clip(nm.x[1], *bounds[1])])
        bd = minimize(obj, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 5000})
        if best is None or bd.fun < best.fun:
            best = bd
    return float(best.fun), float(best.x[0]), float(best.x[1])


def make_figure(z, T, Tpar, vals, cos2, theta, outdir_paper):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping figure.")
        return
    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 10,
                         "font.family": "serif", "axes.linewidth": 0.8})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 3.9))
    fig.subplots_adjust(wspace=0.28)

    ax1.axhline(0, color="0.7", lw=0.6)
    ax1.fill_between(z, Tpar, T, facecolor="#aec4de", edgecolor="none", lw=0,
                     zorder=0, label=r"$T_\perp$: orthogonal to CPL")
    ax1.plot(z, T, "o-", color="#1f4e79", ms=3.6, lw=1.5, label="nuisance template $T(z)$")
    ax1.plot(z, Tpar, "s--", color="#c0392b", ms=2.8, lw=1.4,
             label=r"best CPL fit to $T$  ($T_\parallel$)")
    ax1.set_xlabel("redshift $z$"); ax1.set_ylabel("template amplitude (unit RMS)")
    ax1.set_xlim(0, 1.5); ax1.set_ylim(-2.7, 3.5)
    ax1.legend(frameon=True, framealpha=0.9, edgecolor="0.8", fontsize=7.8,
               loc="lower left", borderpad=0.5)
    ax1.text(0.03, 0.97, f"(a)  $\\theta={theta:.0f}^\\circ$,  "
             f"$\\cos^2\\theta={cos2**2*100:.0f}\\%$ inside CPL",
             transform=ax1.transAxes, fontsize=8.5, va="top")

    labels = ["CPL\nalone", "$T$\nalone", "CPL after $T$\n(adds)", "$T$ after CPL\n(adds)"]
    cols = ["#c0392b", "#1f4e79", "#e08a8a", "#7fa8d0"]
    b = ax2.bar(range(4), vals, color=cols, edgecolor="k", lw=0.6, width=0.66)
    for r, v in zip(b, vals):
        ax2.text(r.get_x() + r.get_width() / 2, v + 0.45, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=9)
    ax2.set_xticks(range(4)); ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel(r"$\Delta\chi^2$ (fixed $\Omega_m$)"); ax2.set_ylim(0, 20.5)
    ax2.text(0.03, 0.96, "(b)", transform=ax2.transAxes, fontsize=9, va="top")
    ax2.annotate("CPL largely\nabsorbed by $T$", xy=(2, 2.4), xytext=(2, 7.0),
                 ha="center", va="bottom", fontsize=8, color="#7a2018",
                 arrowprops=dict(arrowstyle="->", color="#7a2018", lw=0.9))
    ax2.annotate("$T$ has structure\nbeyond CPL", xy=(3, 12.2), xytext=(3, 17.2),
                 ha="center", va="bottom", fontsize=8, color="#16385c",
                 arrowprops=dict(arrowstyle="->", color="#16385c", lw=0.9))

    outdir_paper.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir_paper / "overlap_angle.pdf", bbox_inches="tight")
    fig.savefig(outdir_paper / "overlap_angle.png", dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {outdir_paper/'overlap_angle.pdf'} (+ .png)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/mu_mat_union3_cosmo=2_mu.fits"))
    parser.add_argument("--outdir", type=Path, default=Path("results"))
    parser.add_argument("--paperdir", type=Path, default=Path("paper"))
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()

    z, mu, P = load_union3_compressed(args.data)
    args.outdir.mkdir(parents=True, exist_ok=True)

    T = standardize(gaussian(z, 0.09, 0.06) - gaussian(z, 0.775, 0.10))
    Tlow = standardize(gaussian(z, 0.09, 0.06))
    Tmid = standardize(gaussian(z, 0.775, 0.10))

    mu_l = distance_modulus(z, OMEGA_M_FIXED)
    c_l = gls(mu, P, mu_l)[0]
    c_cpl, w0b, wab = fit_cpl(z, mu, P)
    c_t, beta_t = gls(mu, P, mu_l, T[:, None])
    c_ct, w0n, wan = fit_cpl(z, mu, P, T[:, None])
    a_lcdm = beta_t[1]
    a_cpl = gls(mu, P, distance_modulus(z, OMEGA_M_FIXED, H0_FIXED, w0n, wan), T[:, None])[1][1]

    # ---- overlap angle in the M-marginalized precision metric ----
    one = np.ones((len(z), 1))
    Pone = P @ one
    Peff = P - Pone @ np.linalg.solve(one.T @ Pone, Pone.T)
    inner = lambda a, b: a @ (Peff @ b)
    nrm = lambda a: np.sqrt(inner(a, a))
    dmu = distance_modulus(z, OMEGA_M_FIXED, H0_FIXED, w0b, wab) - mu_l
    cos1 = inner(T, dmu) / (nrm(T) * nrm(dmu))
    eps = 1e-4
    v1 = (distance_modulus(z, OMEGA_M_FIXED, H0_FIXED, w0b + eps, wab)
          - distance_modulus(z, OMEGA_M_FIXED, H0_FIXED, w0b - eps, wab)) / (2 * eps)
    v2 = (distance_modulus(z, OMEGA_M_FIXED, H0_FIXED, w0b, wab + eps)
          - distance_modulus(z, OMEGA_M_FIXED, H0_FIXED, w0b, wab - eps)) / (2 * eps)
    V = np.column_stack([v1, v2])
    Tpar_eff = V @ np.linalg.solve(V.T @ (Peff @ V), V.T @ (Peff @ T))
    cos2 = nrm(Tpar_eff) / nrm(T)
    theta = np.degrees(np.arccos(np.clip(cos2, 0, 1)))
    # projection including the marginalized offset, for plotting alignment
    Vplot = np.column_stack([np.ones(len(z)), v1, v2])
    Tpar_plot = Vplot @ np.linalg.solve(Vplot.T @ (P @ Vplot), Vplot.T @ (P @ T))

    cpl_alone, t_alone = c_l - c_cpl, c_l - c_t
    cpl_after_t, t_after_cpl = c_t - c_ct, c_cpl - c_ct
    shared = cpl_alone + t_alone - (c_l - c_ct)

    overlap = pd.DataFrame([{
        "cos_theta_single": cos1, "cos_theta_subspace": cos2, "theta_deg": theta,
        "frac_inside_cpl": cos2 ** 2, "frac_orthogonal": 1 - cos2 ** 2,
        "cpl_alone_delta_chi2": cpl_alone, "t_alone_delta_chi2": t_alone,
        "cpl_after_t_delta_chi2": cpl_after_t, "t_after_cpl_delta_chi2": t_after_cpl,
        "shared_delta_chi2": shared,
        "amplitude_lcdm_nuisance": a_lcdm, "amplitude_cpl_nuisance": a_cpl,
        "w0_cpl_nuisance": w0n, "wa_cpl_nuisance": wan,
    }])
    overlap.to_csv(args.outdir / "template_cpl_overlap.csv", index=False)

    # ---- component decomposition ----
    def comp_row(name, comp):
        cc = gls(mu, P, mu_l, comp[:, None])[0]
        ccn, _, _ = fit_cpl(z, mu, P, comp[:, None])
        return {"component": name, "delta_chi2_alone": c_l - cc,
                "cpl_after_it": cc - ccn, "it_after_cpl": c_cpl - ccn}
    both = gls(mu, P, mu_l, np.column_stack([Tlow, Tmid]))[0]
    comp = pd.DataFrame([
        comp_row("low_z_arm_G(0.09,0.06)", Tlow),
        comp_row("mid_z_arm_G(0.775,0.10)", Tmid),
        {"component": "both_arms_2_amplitudes", "delta_chi2_alone": c_l - both,
         "cpl_after_it": np.nan, "it_after_cpl": np.nan},
        {"component": "full_template_T", "delta_chi2_alone": t_alone,
         "cpl_after_it": cpl_after_t, "it_after_cpl": t_after_cpl},
    ])
    comp.to_csv(args.outdir / "component_decomposition.csv", index=False)

    print("\nTemplate-CPL overlap (fixed Omega_m = 0.2975)")
    print(f"  cos theta (single)   = {cos1:.3f}")
    print(f"  cos theta (subspace) = {cos2:.3f}   theta = {theta:.1f} deg")
    print(f"  inside CPL = {cos2**2*100:.0f}% ; orthogonal = {(1-cos2**2)*100:.0f}%")
    print(f"  CPL after T = {cpl_after_t:.3f} ; T after CPL = {t_after_cpl:.3f} ; shared = {shared:.3f}")
    print(f"  amplitude a: LCDM+nuis = {a_lcdm:.4f} ; CPL+nuis = {a_cpl:.4f}")
    print("\nComponent decomposition:")
    print(comp.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nSaved: {args.outdir/'template_cpl_overlap.csv'}, {args.outdir/'component_decomposition.csv'}")

    if not args.no_figure:
        make_figure(z, T, Tpar_plot,
                    [cpl_alone, t_alone, cpl_after_t, t_after_cpl], cos2, theta, args.paperdir)


if __name__ == "__main__":
    main()
