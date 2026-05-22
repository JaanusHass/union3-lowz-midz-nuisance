#!/usr/bin/env python
"""
05_three_version_comparison.py

Union3 -> Union3.1 three-version compressed-data comparison.

This script applies the same fixed-Omega_m compressed-data diagnostic to three
public compressed supernova matrices:

1) Union3 original
2) Union3.1 UNITY1.7
3) Union3.1 UNITY1.8

It reproduces the qualitative comparison used in the technical note:
    - Omega_m best-fit shifts downward across versions
    - DESI-like Omega_m penalty decreases
    - low-z / mid-z nuisance amplitude decreases
    - CPL AIC gain becomes small in Union3.1 versions

Required local input files by default:

    data/mu_mat_union3_cosmo=2_mu.fits
    data/mu_mat_union3.1_UNITY1.7_template_cosmo=2_0_mu.fits
    data/mu_mat_union3.1_UNITY1.8_template_cosmo=2_0_mu.fits

The data files are public Union3 / Union3.1 release products and are not
redistributed here by default.

Run from repository root:
    python scripts/05_three_version_comparison.py

This is a compressed-data diagnostic only. It is not the official
DESI/CMB/Union3 full likelihood.
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
H0_FIXED = 70.0
OMEGA_M_DESI_LIKE = 0.2975


DEFAULT_FILES = {
    "Union3 original": Path("data/mu_mat_union3_cosmo=2_mu.fits"),
    "Union3.1 UNITY1.7": Path("data/mu_mat_union3.1_UNITY1.7_template_cosmo=2_0_mu.fits"),
    "Union3.1 UNITY1.8": Path("data/mu_mat_union3.1_UNITY1.8_template_cosmo=2_0_mu.fits"),
}


def load_compressed_matrix(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Download the public compressed FITS file "
            "from rubind/union3_release and place it in data/."
        )

    with fits.open(path) as hdul:
        mat = np.asarray(hdul[0].data, dtype=float)

    z = np.asarray(mat[0, 1:], dtype=float)
    mu = np.asarray(mat[1:, 0], dtype=float)
    precision = np.asarray(mat[1:, 1:], dtype=float)

    if len(z) != len(mu) or precision.shape != (len(z), len(z)):
        raise ValueError(
            f"Unexpected matrix format for {path}: z={len(z)}, mu={len(mu)}, P={precision.shape}"
        )

    order = np.argsort(z)
    return z[order], mu[order], precision[np.ix_(order, order)]


def e_z(z: np.ndarray, omega_m: float, w0: float = -1.0, wa: float = 0.0) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    de = (1.0 + z) ** (3.0 * (1.0 + w0 + wa)) * np.exp(-3.0 * wa * z / (1.0 + z))
    ez2 = omega_m * (1.0 + z) ** 3 + (1.0 - omega_m) * de
    if np.any(ez2 <= 0.0) or not np.all(np.isfinite(ez2)):
        return np.full_like(z, np.nan)
    return np.sqrt(ez2)


def distance_modulus(
    z: np.ndarray,
    omega_m: float,
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


def low_minus_mid_step_template(z: np.ndarray) -> np.ndarray:
    """
    Simple step-like comparison template used as a functional-form cross-check.
    Low-z is positive and mid-z is negative, then standardized.
    """
    t = np.zeros_like(z, dtype=float)
    t[(z >= 0.00) & (z < 0.20)] = 1.0
    t[(z >= 0.55) & (z < 1.00)] = -1.0
    return standardize(t)


def gls_chi2(
    mu_obs: np.ndarray,
    precision: np.ndarray,
    mu_model: np.ndarray,
    template: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray]:
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


def lcdm_chi2_at_omega(
    z: np.ndarray,
    mu_obs: np.ndarray,
    precision: np.ndarray,
    omega_m: float,
    template: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray]:
    mu_model = distance_modulus(z, omega_m=omega_m, w0=-1.0, wa=0.0)
    return gls_chi2(mu_obs, precision, mu_model, template)


def fit_lcdm_omega(z: np.ndarray, mu_obs: np.ndarray, precision: np.ndarray) -> Tuple[float, float]:
    def objective(x: np.ndarray) -> float:
        omega_m = float(x[0])
        chi2, _ = lcdm_chi2_at_omega(z, mu_obs, precision, omega_m)
        return chi2

    res = minimize(objective, np.array([0.33]), method="L-BFGS-B", bounds=[(0.05, 0.8)])
    return float(res.x[0]), float(res.fun)


def fit_cpl_at_fixed_omega(
    z: np.ndarray,
    mu_obs: np.ndarray,
    precision: np.ndarray,
    omega_m: float = OMEGA_M_DESI_LIKE,
    template: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
    def objective(x: np.ndarray) -> float:
        w0, wa = float(x[0]), float(x[1])
        mu_model = distance_modulus(z, omega_m=omega_m, w0=w0, wa=wa)
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


def analyse_version(label: str, path: Path) -> dict:
    z, mu, precision = load_compressed_matrix(path)

    omega_best, chi2_best = fit_lcdm_omega(z, mu, precision)
    chi2_desi, _ = lcdm_chi2_at_omega(z, mu, precision, OMEGA_M_DESI_LIKE)

    cpl_chi2, cpl_w0, cpl_wa = fit_cpl_at_fixed_omega(z, mu, precision, OMEGA_M_DESI_LIKE, template=None)

    t_gauss = low_mid_template(z)
    chi2_gauss, beta_gauss = lcdm_chi2_at_omega(z, mu, precision, OMEGA_M_DESI_LIKE, template=t_gauss)

    t_step = low_minus_mid_step_template(z)
    chi2_step, beta_step = lcdm_chi2_at_omega(z, mu, precision, OMEGA_M_DESI_LIKE, template=t_step)

    cpl_after_gauss_chi2, cpl_after_gauss_w0, cpl_after_gauss_wa = fit_cpl_at_fixed_omega(
        z, mu, precision, OMEGA_M_DESI_LIKE, template=t_gauss
    )
    cpl_after_step_chi2, cpl_after_step_w0, cpl_after_step_wa = fit_cpl_at_fixed_omega(
        z, mu, precision, OMEGA_M_DESI_LIKE, template=t_step
    )

    cpl_delta = chi2_desi - cpl_chi2
    gauss_delta = chi2_desi - chi2_gauss
    step_delta = chi2_desi - chi2_step
    cpl_after_gauss_delta = chi2_gauss - cpl_after_gauss_chi2
    cpl_after_step_delta = chi2_step - cpl_after_step_chi2

    return {
        "version": label,
        "file": str(path),
        "n_nodes": len(z),
        "omega_m_best_lcdm": omega_best,
        "lcdm_best_chi2": chi2_best,
        "lcdm_fixed_omega_chi2": chi2_desi,
        "desi_like_omega_penalty_delta_chi2": chi2_desi - chi2_best,
        "cpl_delta_chi2": cpl_delta,
        "cpl_aic_gain": cpl_delta - 4.0,
        "cpl_w0": cpl_w0,
        "cpl_wa": cpl_wa,
        "low_mid_gaussian_delta_chi2": gauss_delta,
        "low_mid_gaussian_aic_gain": gauss_delta - 2.0,
        "low_mid_gaussian_amplitude_mag": float(beta_gauss[1]),
        "low_mid_step_delta_chi2": step_delta,
        "low_mid_step_aic_gain": step_delta - 2.0,
        "low_mid_step_amplitude_mag": float(beta_step[1]),
        "cpl_after_gaussian_delta_chi2": cpl_after_gauss_delta,
        "cpl_after_gaussian_aic_gain": cpl_after_gauss_delta - 4.0,
        "cpl_after_gaussian_w0": cpl_after_gauss_w0,
        "cpl_after_gaussian_wa": cpl_after_gauss_wa,
        "cpl_after_step_delta_chi2": cpl_after_step_delta,
        "cpl_after_step_aic_gain": cpl_after_step_delta - 4.0,
        "cpl_after_step_w0": cpl_after_step_w0,
        "cpl_after_step_wa": cpl_after_step_wa,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("results"))
    parser.add_argument("--union3", type=Path, default=DEFAULT_FILES["Union3 original"])
    parser.add_argument("--unity17", type=Path, default=DEFAULT_FILES["Union3.1 UNITY1.7"])
    parser.add_argument("--unity18", type=Path, default=DEFAULT_FILES["Union3.1 UNITY1.8"])
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    files = {
        "Union3 original": args.union3,
        "Union3.1 UNITY1.7": args.unity17,
        "Union3.1 UNITY1.8": args.unity18,
    }

    rows = []
    missing = []
    for label, path in files.items():
        if not path.exists():
            missing.append((label, path))
        else:
            rows.append(analyse_version(label, path))

    if missing:
        print("\nMissing required data files:")
        for label, path in missing:
            print(f"  - {label}: {path}")
        print("\nDownload the public compressed FITS files from rubind/union3_release and place them in data/.")
        raise SystemExit(1)

    df = pd.DataFrame(rows)
    outpath = args.outdir / "three_version_comparison.csv"
    df.to_csv(outpath, index=False)

    view_cols = [
        "version",
        "omega_m_best_lcdm",
        "desi_like_omega_penalty_delta_chi2",
        "cpl_delta_chi2",
        "cpl_aic_gain",
        "low_mid_gaussian_delta_chi2",
        "low_mid_gaussian_aic_gain",
        "low_mid_gaussian_amplitude_mag",
        "cpl_after_gaussian_aic_gain",
    ]

    print("\nUnion3 -> Union3.1 three-version comparison")
    print(df[view_cols].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nSaved results to: {outpath}")


if __name__ == "__main__":
    main()
