#!/usr/bin/env python
"""
08_orthogonalized_null.py

Orthogonalized (beyond-CPL) null test for the Union3 compressed-data low-z / mid-z
nuisance diagnostic.

The standard random null test (03_random_template_null.py) compares the STANDALONE
improvement of each template. This script adds the sharper quantity that matters for the
degeneracy argument (Section 7.6): the improvement a template adds AFTER CPL is already
in the model,

    Delta chi2_after_CPL = chi2(CPL) - chi2(CPL + template),

which removes the part of the residual CPL can itself absorb. It compares the original
low-z / mid-z template against the same 200 random smooth templates, using the identical
random-template recipe and draw order as 03_random_template_null.py (so the standalone
column reproduces that script's distribution: max ~12.43, 95th ~9.0).

Expected (fixed-Omega_m = 0.2975 protocol):
    - original template, standalone Delta chi2 ~ 17.12
    - original template, after-CPL Delta chi2 ~ 10.45
    - 0/200 random templates reach the original after-CPL value

Output:
    results/orthogonalized_null.csv

Required input file:
    data/mu_mat_union3_cosmo=2_mu.fits

Run from repository root:
    python scripts/08_orthogonalized_null.py
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


def e_z(z: np.ndarray, omega_m: float, w0: float = -1.0, wa: float = 0.0) -> np.ndarray:
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
    rms = np.sqrt(np.mean(y ** 2))
    if rms <= 0:
        raise ValueError("Zero-RMS template")
    return y / rms


def primary_template(z):
    return standardize(gaussian(z, 0.09, 0.06) - gaussian(z, 0.775, 0.10))


def random_template(z, rng):
    # Same recipe and DRAW ORDER as 03_random_template_null.py, so the standalone
    # column of this script reproduces that script's random distribution.
    c_low = rng.uniform(0.05, 0.40)
    c_mid = rng.uniform(0.40, 1.50)
    s_low = rng.uniform(0.04, 0.20)
    s_mid = rng.uniform(0.04, 0.20)
    sign = rng.choice([-1.0, 1.0])
    t = sign * (gaussian(z, c_low, s_low) - gaussian(z, c_mid, s_mid))
    return standardize(t)


def gls_chi2(mu_obs, precision, mu_model, template=None):
    if np.any(~np.isfinite(mu_model)):
        return np.inf
    residual = mu_obs - mu_model
    if template is None:
        xmat = np.ones((len(mu_obs), 1))
    else:
        xmat = np.column_stack([np.ones(len(mu_obs)), template])
    fisher = xmat.T @ precision @ xmat
    rhs = xmat.T @ precision @ residual
    beta = np.linalg.pinv(fisher) @ rhs
    final = residual - xmat @ beta
    return float(final.T @ precision @ final)


def fit_cpl_chi2(z, mu_obs, precision, template=None):
    """CPL chi2 at fixed Omega_m; Nelder-Mead seed then bounded L-BFGS-B (as scripts 01/02)."""
    def objective(x):
        return gls_chi2(mu_obs, precision,
                        distance_modulus(z, OMEGA_M_FIXED, H0_FIXED, float(x[0]), float(x[1])),
                        template)
    bounds = [(-3.0, 1.0), (-5.0, 5.0)]
    starts = [(-1.0, 0.0), (-0.7, -1.0), (-1.2, 1.0), (-0.5, -2.0), (-1.5, 2.0)]
    best = np.inf
    for s in starts:
        nm = minimize(objective, np.array(s), method="Nelder-Mead", options={"maxiter": 5000})
        x0 = np.array([np.clip(nm.x[0], *bounds[0]), np.clip(nm.x[1], *bounds[1])])
        bd = minimize(objective, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 5000})
        if np.isfinite(bd.fun) and float(bd.fun) < best:
            best = float(bd.fun)
    if not np.isfinite(best):
        raise RuntimeError("CPL fit failed.")
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/mu_mat_union3_cosmo=2_mu.fits"))
    parser.add_argument("--outdir", type=Path, default=Path("results"))
    parser.add_argument("--n-random", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    z, mu_obs, precision = load_union3_compressed(args.data)
    args.outdir.mkdir(parents=True, exist_ok=True)

    mu_lcdm = distance_modulus(z, OMEGA_M_FIXED, H0_FIXED, -1.0, 0.0)
    chi2_lcdm = gls_chi2(mu_obs, precision, mu_lcdm)
    chi2_cpl = fit_cpl_chi2(z, mu_obs, precision, template=None)

    def standalone(t):
        return chi2_lcdm - gls_chi2(mu_obs, precision, mu_lcdm, t)

    def after_cpl(t):
        return chi2_cpl - fit_cpl_chi2(z, mu_obs, precision, template=t)

    T = primary_template(z)
    orig_standalone = standalone(T)
    orig_after = after_cpl(T)

    rng = np.random.default_rng(args.seed)
    rows = []
    for i in range(args.n_random):
        t = random_template(z, rng)
        rows.append({
            "index": i,
            "standalone_delta_chi2": standalone(t),
            "after_cpl_delta_chi2": after_cpl(t),
        })
    df = pd.DataFrame(rows)
    out = args.outdir / "orthogonalized_null.csv"
    df.to_csv(out, index=False)

    sa, ac = df["standalone_delta_chi2"], df["after_cpl_delta_chi2"]
    n_ge = int((ac >= orig_after).sum())

    print("\nOrthogonalized (beyond-CPL) null test")
    print(f"Data file: {args.data}")
    print(f"N random templates: {args.n_random}   seed: {args.seed}")
    print(f"\nOriginal template:  standalone Delta chi2 = {orig_standalone:.4f}   "
          f"after-CPL Delta chi2 = {orig_after:.4f}")
    print("\nRandom STANDALONE (reproduces 03_random_template_null.py):")
    print(f"  median = {sa.median():.4f}   95th = {sa.quantile(0.95):.4f}   max = {sa.max():.4f}")
    print("\nRandom AFTER-CPL (beyond-CPL structure):")
    print(f"  median = {ac.median():.4f}   95th = {ac.quantile(0.95):.4f}   max = {ac.max():.4f}")
    print(f"  random >= original after-CPL: {n_ge}/{args.n_random}   "
          f"add-one p = {(n_ge + 1) / (args.n_random + 1):.4f}")
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
