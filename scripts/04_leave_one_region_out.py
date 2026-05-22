#!/usr/bin/env python
"""
04_leave_one_region_out.py

Leave-one-redshift-region-out robustness test for the Union3 compressed-data
low-z / mid-z nuisance diagnostic.

This script removes broad redshift regions one at a time and repeats the
compressed-data four-model diagnostic:

    - LambdaCDM
    - CPL
    - LambdaCDM + nuisance
    - CPL + nuisance

Important technical point:
    For node-removal tests, the covariance matrix C = P^-1 is subselected
    and then re-inverted. Do not directly subset the precision matrix.

Required input file:
    data/mu_mat_union3_cosmo=2_mu.fits

Run from repository root:
    python scripts/04_leave_one_region_out.py
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


def gaussian(z: np.ndarray, c: float, s: float) -> np.ndarray:
    return np.exp(-0.5 * ((z - c) / s) ** 2)


def standardize(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=float) - np.mean(x)
    rms = np.sqrt(np.mean(y**2))
    if rms <= 0:
        raise ValueError("Zero-RMS template")
    return y / rms


def low_mid_template(z: np.ndarray) -> np.ndarray:
    return standardize(gaussian(z, 0.09, 0.06) - gaussian(z, 0.775, 0.10))


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


def fit_lcdm_chi2(
    z: np.ndarray,
    mu_obs: np.ndarray,
    precision: np.ndarray,
    template: Optional[np.ndarray] = None,
) -> tuple[float, np.ndarray]:
    mu_model = distance_modulus(z, OMEGA_M_FIXED, H0_FIXED, -1.0, 0.0)
    return gls_chi2(mu_obs, precision, mu_model, template)


def fit_cpl(
    z: np.ndarray,
    mu_obs: np.ndarray,
    precision: np.ndarray,
    template: Optional[np.ndarray] = None,
) -> tuple[float, float, float]:
    def objective(x: np.ndarray) -> float:
        w0, wa = float(x[0]), float(x[1])
        mu_model = distance_modulus(z, OMEGA_M_FIXED, H0_FIXED, w0, wa)
        chi2, _ = gls_chi2(mu_obs, precision, mu_model, template)
        return chi2

    bounds = [(-3.0, 1.0), (-5.0, 5.0)]
    starts = [
        np.array([-1.0, 0.0]),
        np.array([-0.7, -1.0]),
        np.array([-1.2, 1.0]),
        np.array([-0.5, -2.0]),
        np.array([-1.5, 2.0]),
    ]

    best_fun = np.inf
    best_x = np.array([np.nan, np.nan])
    for start in starts:
        res = minimize(objective, start, method="Nelder-Mead", options={"maxiter": 5000})
        x0 = np.array([np.clip(res.x[0], -3.0, 1.0), np.clip(res.x[1], -5.0, 5.0)])
        res2 = minimize(objective, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 5000})

        for cand in [res, res2]:
            if float(cand.fun) < best_fun:
                best_fun = float(cand.fun)
                best_x = np.asarray(cand.x, dtype=float)

    return best_fun, float(best_x[0]), float(best_x[1])


def run_case(label: str, zmin: Optional[float], zmax: Optional[float], z: np.ndarray, mu: np.ndarray, precision: np.ndarray) -> dict:
    if zmin is None and zmax is None:
        mask = np.ones_like(z, dtype=bool)
        removed = "none"
    elif zmin is None:
        mask = ~(z < zmax)
        removed = f"z < {zmax}"
    elif zmax is None:
        mask = ~(z >= zmin)
        removed = f"z >= {zmin}"
    else:
        mask = ~((z >= zmin) & (z < zmax))
        removed = f"{zmin}-{zmax}"

    z_sub = z[mask]
    mu_sub = mu[mask]
    precision_sub = subselect_precision(precision, mask)
    template = low_mid_template(z_sub)

    lcdm_chi2, _ = fit_lcdm_chi2(z_sub, mu_sub, precision_sub, template=None)
    cpl_chi2, cpl_w0, cpl_wa = fit_cpl(z_sub, mu_sub, precision_sub, template=None)
    nuis_chi2, beta = fit_lcdm_chi2(z_sub, mu_sub, precision_sub, template=template)
    cpl_nuis_chi2, cpl_nuis_w0, cpl_nuis_wa = fit_cpl(z_sub, mu_sub, precision_sub, template=template)

    return {
        "case": label,
        "removed_z": removed,
        "n_kept": int(mask.sum()),
        "lcdm_chi2": lcdm_chi2,
        "cpl_chi2": cpl_chi2,
        "nuisance_chi2": nuis_chi2,
        "cpl_nuisance_chi2": cpl_nuis_chi2,
        "delta_chi2_cpl": lcdm_chi2 - cpl_chi2,
        "delta_chi2_nuisance": lcdm_chi2 - nuis_chi2,
        "delta_chi2_cpl_after_nuisance": nuis_chi2 - cpl_nuis_chi2,
        "nuisance_amplitude": float(beta[1]),
        "cpl_w0": cpl_w0,
        "cpl_wa": cpl_wa,
        "cpl_nuisance_w0": cpl_nuis_w0,
        "cpl_nuisance_wa": cpl_nuis_wa,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/mu_mat_union3_cosmo=2_mu.fits"))
    parser.add_argument("--outdir", type=Path, default=Path("results"))
    args = parser.parse_args()

    z, mu, precision = load_union3_compressed(args.data)
    args.outdir.mkdir(parents=True, exist_ok=True)

    cases = [
        ("Full sample", None, None),
        ("Very low z", 0.00, 0.10),
        ("Low z", 0.10, 0.30),
        ("Mid z", 0.30, 0.70),
        ("High-mid z", 0.70, 1.00),
        ("High z", 1.00, None),
    ]

    rows = [run_case(label, zmin, zmax, z, mu, precision) for label, zmin, zmax in cases]
    df = pd.DataFrame(rows)

    outpath = args.outdir / "leave_one_region_out.csv"
    df.to_csv(outpath, index=False)

    print("\nLeave-one-redshift-region-out robustness test")
    print(f"Data file: {args.data}")
    print("\nKey columns:")
    print(
        df[
            [
                "case",
                "removed_z",
                "n_kept",
                "delta_chi2_cpl",
                "delta_chi2_nuisance",
                "delta_chi2_cpl_after_nuisance",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )
    print(f"\nSaved results to: {outpath}")


if __name__ == "__main__":
    main()
