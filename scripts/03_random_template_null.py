#!/usr/bin/env python
"""
03_random_template_null.py

Random smooth-template null test for the Union3 compressed-data low-z / mid-z
nuisance diagnostic.

This script compares the original low-z / mid-z nuisance template with random
smooth two-Gaussian templates.

Primary template:
    T(z) = G(z; 0.09, 0.06) - G(z; 0.775, 0.10)

Random templates:
    - difference of two Gaussians
    - low-z center uniformly from [0.05, 0.40]
    - mid/high-z center uniformly from [0.40, 1.50]
    - widths uniformly from [0.04, 0.20]
    - random sign
    - standardized to zero mean and unit RMS on the 22 Union3 nodes

Output:
    results/random_template_null.csv

Important:
    This script does not write a separate summary CSV. Summary statistics are
    printed to the terminal from the same in-memory table that is written to
    random_template_null.csv. This avoids stale summary-file mismatches.

Required input file:
    data/mu_mat_union3_cosmo=2_mu.fits

Run from repository root:
    python scripts/03_random_template_null.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid

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


def distance_modulus(z: np.ndarray, omega_m: float = OMEGA_M_FIXED, h0: float = H0_FIXED) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    grid = np.linspace(0.0, float(np.max(z)) * 1.001 + 1e-6, 6000)
    ez = e_z(grid, omega_m=omega_m, w0=-1.0, wa=0.0)
    if np.any(~np.isfinite(ez)):
        return np.full_like(z, np.nan)

    integral = cumulative_trapezoid(1.0 / ez, grid, initial=0.0)
    chi = np.interp(z, grid, integral)
    dl_mpc = (C_KM_S / h0) * (1.0 + z) * chi
    if np.any(dl_mpc <= 0.0):
        return np.full_like(z, np.nan)
    return 5.0 * np.log10(dl_mpc) + 25.0


def gaussian(z: np.ndarray, c: float, s: float) -> np.ndarray:
    return np.exp(-0.5 * ((z - c) / s) ** 2)


def standardize(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=float) - np.mean(x)
    rms = np.sqrt(np.mean(y**2))
    if rms <= 0:
        raise ValueError("Zero-RMS template")
    return y / rms


def primary_template(z: np.ndarray) -> np.ndarray:
    return standardize(gaussian(z, 0.09, 0.06) - gaussian(z, 0.775, 0.10))


def random_template(z: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    c_low = rng.uniform(0.05, 0.40)
    c_mid = rng.uniform(0.40, 1.50)
    s_low = rng.uniform(0.04, 0.20)
    s_mid = rng.uniform(0.04, 0.20)
    sign = rng.choice([-1.0, 1.0])

    t = sign * (gaussian(z, c_low, s_low) - gaussian(z, c_mid, s_mid))
    return standardize(t), {
        "c_low": c_low,
        "c_mid": c_mid,
        "s_low": s_low,
        "s_mid": s_mid,
        "sign": sign,
    }


def gls_chi2(
    mu_obs: np.ndarray,
    precision: np.ndarray,
    mu_model: np.ndarray,
    template: Optional[np.ndarray] = None,
) -> tuple[float, np.ndarray]:
    if np.any(~np.isfinite(mu_model)):
        return np.inf, np.array([np.nan])

    residual = mu_obs - mu_model
    if template is None:
        xmat = np.ones((len(mu_obs), 1))
    else:
        xmat = np.column_stack([np.ones(len(mu_obs)), template])

    fisher = xmat.T @ precision @ xmat
    rhs = xmat.T @ precision @ residual
    beta = np.linalg.pinv(fisher) @ rhs
    final = residual - xmat @ beta
    return float(final.T @ precision @ final), beta


def nuisance_delta_chi2(
    mu_obs: np.ndarray,
    precision: np.ndarray,
    mu_model: np.ndarray,
    template: np.ndarray,
) -> tuple[float, float]:
    chi2_lcdm, _ = gls_chi2(mu_obs, precision, mu_model, template=None)
    chi2_nuis, beta = gls_chi2(mu_obs, precision, mu_model, template=template)
    return chi2_lcdm - chi2_nuis, float(beta[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/mu_mat_union3_cosmo=2_mu.fits"))
    parser.add_argument("--outdir", type=Path, default=Path("results"))
    parser.add_argument("--n-random", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    z, mu_obs, precision = load_union3_compressed(args.data)
    mu_model = distance_modulus(z)

    args.outdir.mkdir(parents=True, exist_ok=True)

    original_delta, original_amp = nuisance_delta_chi2(
        mu_obs, precision, mu_model, primary_template(z)
    )

    rng = np.random.default_rng(args.seed)
    rows = []

    for i in range(args.n_random):
        t, meta = random_template(z, rng)
        delta, amp = nuisance_delta_chi2(mu_obs, precision, mu_model, t)
        rows.append(
            {
                "index": i,
                **meta,
                "delta_chi2": delta,
                "amplitude": amp,
                "exceeds_or_equals_original": bool(delta >= original_delta),
                "strictly_exceeds_original": bool(delta > original_delta),
            }
        )

    df = pd.DataFrame(rows)
    random_path = args.outdir / "random_template_null.csv"
    df.to_csv(random_path, index=False)

    n_ge = int((df["delta_chi2"] >= original_delta).sum())
    n_gt = int((df["delta_chi2"] > original_delta).sum())
    imax = int(df["delta_chi2"].idxmax())
    closest = df.loc[imax]

    print("\nRandom smooth-template null test")
    print(f"Data file: {args.data}")
    print(f"N random templates: {args.n_random}")
    print(f"Random seed: {args.seed}")
    print(f"Original template Delta chi2: {original_delta:.6f}")
    print(f"Original template amplitude: {original_amp:.6f}")
    print(f"Random median Delta chi2: {df['delta_chi2'].median():.6f}")
    print(f"Random 95th percentile Delta chi2: {df['delta_chi2'].quantile(0.95):.6f}")
    print(f"Random max Delta chi2: {df['delta_chi2'].max():.6f}")
    print(f"Random templates >= original: {n_ge}/{args.n_random}")
    print(f"Random templates > original: {n_gt}/{args.n_random}")
    print(f"Empirical tail bound: p < 1/{args.n_random + 1}" if n_ge == 0 else f"Empirical tail: {n_ge}/{args.n_random}")
    print("\nClosest random template:")
    print(f"  index = {int(closest['index'])}")
    print(f"  delta_chi2 = {closest['delta_chi2']:.6f}")
    print(f"  c_low = {closest['c_low']:.6f}")
    print(f"  c_mid = {closest['c_mid']:.6f}")
    print(f"  s_low = {closest['s_low']:.6f}")
    print(f"  s_mid = {closest['s_mid']:.6f}")
    print(f"  sign = {closest['sign']:.0f}")

    print(f"\nSaved random-template results to: {random_path}")


if __name__ == "__main__":
    main()
