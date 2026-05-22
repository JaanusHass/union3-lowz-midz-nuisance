#!/usr/bin/env python
"""
06_cumulative_lowz_removal.py

Cumulative low-z node-removal diagnostic for the Union3 compressed-data
low-z / mid-z nuisance question.

This script directly addresses the concern:

    "Is the low-z Gaussian nuisance just acting like removing the low-z SNe?"

It removes cumulative low-z ranges and refits:

    - LambdaCDM
    - CPL

The intended comparison is to check how the CPL improvement changes when
low-redshift nodes are removed:

    - full sample
    - remove z < 0.08
    - remove z < 0.10
    - remove z < 0.12
    - remove z < 0.20
    - remove z < 0.30

Important technical point:
    For node-removal tests, the covariance matrix C = P^-1 is subselected
    and then re-inverted. Do not directly subset the precision matrix.

Required input file:
    data/mu_mat_union3_cosmo=2_mu.fits

Run from repository root:
    python scripts/06_cumulative_lowz_removal.py

This is a compressed-data diagnostic only. It is not the official
DESI/CMB/Union3 full likelihood.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

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


def subselect_precision(precision: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Correct node-removal operation:
    convert precision to covariance, subselect covariance, then invert.
    """
    covariance = np.linalg.inv(precision)
    covariance_sub = covariance[np.ix_(mask, mask)]
    return np.linalg.inv(covariance_sub)


def e_z(z: np.ndarray, omega_m: float, w0: float = -1.0, wa: float = 0.0) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    de = (1.0 + z) ** (3.0 * (1.0 + w0 + wa)) * np.exp(-3.0 * wa * z / (1.0 + z))
    ez2 = omega_m * (1.0 + z) ** 3 + (1.0 - omega_m) * de
    if np.any(ez2 <= 0.0) or not np.all(np.isfinite(ez2)):
        return np.full_like(z, np.nan)
    return np.sqrt(ez2)


def distance_modulus(
    z: np.ndarray,
    omega_m: float = OMEGA_M_FIXED,
    h0: float = H0_FIXED,
    w0: float = -1.0,
    wa: float = 0.0,
) -> np.ndarray:
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


def gls_chi2(mu_obs: np.ndarray, precision: np.ndarray, mu_model: np.ndarray) -> tuple[float, float]:
    """
    Generalized least squares fit for the intercept M only.

    Model:
        mu_obs = mu_model + M
    """
    residual = mu_obs - mu_model
    xmat = np.ones((len(mu_obs), 1))
    fisher = xmat.T @ precision @ xmat
    rhs = xmat.T @ precision @ residual
    beta = np.linalg.pinv(fisher) @ rhs
    final = residual - xmat @ beta
    return float(final.T @ precision @ final), float(beta[0])


def lcdm_chi2(z: np.ndarray, mu_obs: np.ndarray, precision: np.ndarray) -> tuple[float, float]:
    mu_model = distance_modulus(z, OMEGA_M_FIXED, H0_FIXED, -1.0, 0.0)
    return gls_chi2(mu_obs, precision, mu_model)


def cpl_fit(z: np.ndarray, mu_obs: np.ndarray, precision: np.ndarray) -> tuple[float, float, float, float]:
    """
    Fit CPL chi2 at fixed Omega_m.

    Nelder-Mead is used only to find a good starting point.
    The accepted final result is always the bounded L-BFGS-B result,
    so w0 and wa remain inside the declared parameter bounds.
    """

    def objective(x: np.ndarray) -> float:
        w0, wa = float(x[0]), float(x[1])
        mu_model = distance_modulus(z, OMEGA_M_FIXED, H0_FIXED, w0, wa)
        chi2, _ = gls_chi2(mu_obs, precision, mu_model)
        return chi2

    bounds = [(-3.0, 1.0), (-5.0, 5.0)]
    starts = [
        np.array([-1.0, 0.0]),
        np.array([-0.7, -1.0]),
        np.array([-1.2, 1.0]),
        np.array([-0.5, -2.0]),
        np.array([-1.5, 2.0]),
    ]

    best = None

    for start in starts:
        # Use Nelder-Mead only as a preliminary unconstrained search.
        res_nm = minimize(
            objective,
            start,
            method="Nelder-Mead",
            options={"maxiter": 5000},
        )

        # Clip the preliminary result back into the allowed parameter range.
        x0 = np.array(
            [
                np.clip(res_nm.x[0], bounds[0][0], bounds[0][1]),
                np.clip(res_nm.x[1], bounds[1][0], bounds[1][1]),
            ]
        )

        # Final accepted result must respect the bounds.
        res_bounded = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 5000},
        )

        if best is None or res_bounded.fun < best.fun:
            best = res_bounded

    if best is None or not best.success:
        raise RuntimeError("CPL fit failed to converge with bounded optimizer.")

    w0, wa = float(best.x[0]), float(best.x[1])
    mu_model = distance_modulus(z, OMEGA_M_FIXED, H0_FIXED, w0, wa)
    chi2, M = gls_chi2(mu_obs, precision, mu_model)

    return chi2, w0, wa, M


def run_case(label: str, threshold: float | None, z: np.ndarray, mu: np.ndarray, precision: np.ndarray) -> dict:
    if threshold is None:
        mask = np.ones_like(z, dtype=bool)
        removed = "none"
    else:
        mask = z >= threshold
        removed = f"z < {threshold:.2f}"

    z_sub = z[mask]
    mu_sub = mu[mask]
    precision_sub = subselect_precision(precision, mask)

    lcdm, M_lcdm = lcdm_chi2(z_sub, mu_sub, precision_sub)
    cpl, w0, wa, M_cpl = cpl_fit(z_sub, mu_sub, precision_sub)

    delta = lcdm - cpl

    return {
        "case": label,
        "removed": removed,
        "z_threshold": np.nan if threshold is None else threshold,
        "n_kept": int(mask.sum()),
        "n_removed": int((~mask).sum()),
        "lcdm_chi2": lcdm,
        "cpl_chi2": cpl,
        "cpl_delta_chi2": delta,
        "cpl_aic_gain": delta - 4.0,
        "cpl_w0": w0,
        "cpl_wa": wa,
        "M_lcdm": M_lcdm,
        "M_cpl": M_cpl,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/mu_mat_union3_cosmo=2_mu.fits"))
    parser.add_argument("--outdir", type=Path, default=Path("results"))
    args = parser.parse_args()

    z, mu, precision = load_union3_compressed(args.data)
    args.outdir.mkdir(parents=True, exist_ok=True)

    thresholds = [
        ("Full sample", None),
        ("Remove z < 0.08", 0.08),
        ("Remove z < 0.10", 0.10),
        ("Remove z < 0.12", 0.12),
        ("Remove z < 0.20", 0.20),
        ("Remove z < 0.30", 0.30),
    ]

    rows = [run_case(label, threshold, z, mu, precision) for label, threshold in thresholds]
    df = pd.DataFrame(rows)

    outpath = args.outdir / "cumulative_lowz_removal.csv"
    df.to_csv(outpath, index=False)

    print("\nCumulative low-z removal diagnostic")
    print(f"Data file: {args.data}")
    print(
        df[
            [
                "case",
                "removed",
                "n_kept",
                "n_removed",
                "cpl_delta_chi2",
                "cpl_aic_gain",
                "cpl_w0",
                "cpl_wa",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )
    print(f"\nSaved results to: {outpath}")


if __name__ == "__main__":
    main()