#!/usr/bin/env python
"""
01_union3_fixed_omega_test.py

Reproduces the core fixed-Omega_m Union3 compressed-data diagnostic from:

Hass 2026, "A Low-z / Mid-z Nuisance Mode in the Union3 Compressed Supernova Data"

Expected approximate results with the public Union3 file mu_mat_union3_cosmo=2_mu.fits:

- LambdaCDM at fixed Omega_m = 0.2975
- CPL improvement vs LambdaCDM: Delta chi2 ≈ 7.536
- Low-z / mid-z Gaussian nuisance improvement vs LambdaCDM: Delta chi2 ≈ 17.120
- CPL after nuisance: Delta chi2 ≈ 0.870

This is a compressed-data robustness diagnostic only. It is not the official
DESI/CMB/Union3 full likelihood.

Required input file:
    mu_mat_union3_cosmo=2_mu.fits

Run from the repository root:
    python scripts/01_union3_fixed_omega_test.py

Or provide the data path:
    python scripts/01_union3_fixed_omega_test.py --data data/mu_mat_union3_cosmo=2_mu.fits
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize

try:
    from astropy.io import fits
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: astropy. Install with: pip install astropy"
    ) from exc


C_KM_S = 299792.458
OMEGA_M_FIXED = 0.2975
H0_FIXED = 70.0


@dataclass
class FitResult:
    model: str
    chi2: float
    params: dict
    k_extra: int


def load_union3_compressed(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load Union3 compressed nodes, distance moduli, and precision matrix."""
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Download mu_mat_union3_cosmo=2_mu.fits "
            "from rubind/union3_release and place it in data/ or repository root."
        )

    with fits.open(path) as hdul:
        mat = np.asarray(hdul[0].data, dtype=float)

    z = np.asarray(mat[0, 1:], dtype=float)
    mu = np.asarray(mat[1:, 0], dtype=float)
    precision = np.asarray(mat[1:, 1:], dtype=float)

    if len(z) != len(mu):
        raise ValueError(f"Unexpected Union3 shape: len(z)={len(z)}, len(mu)={len(mu)}")
    if precision.shape != (len(z), len(z)):
        raise ValueError(
            f"Unexpected precision shape: {precision.shape}, expected {(len(z), len(z))}"
        )

    order = np.argsort(z)
    return z[order], mu[order], precision[np.ix_(order, order)]


def e_z(z: np.ndarray, omega_m: float, w0: float = -1.0, wa: float = 0.0) -> np.ndarray:
    """Flat CPL expansion function E(z)."""
    z = np.asarray(z, dtype=float)
    de = (1.0 + z) ** (3.0 * (1.0 + w0 + wa)) * np.exp(
        -3.0 * wa * z / (1.0 + z)
    )
    ez2 = omega_m * (1.0 + z) ** 3 + (1.0 - omega_m) * de
    if np.any(ez2 <= 0.0) or not np.all(np.isfinite(ez2)):
        return np.full_like(z, np.nan, dtype=float)
    return np.sqrt(ez2)


def distance_modulus(
    z: np.ndarray,
    omega_m: float = OMEGA_M_FIXED,
    h0: float = H0_FIXED,
    w0: float = -1.0,
    wa: float = 0.0,
) -> np.ndarray:
    """Distance modulus for flat CPL cosmology."""
    z = np.asarray(z, dtype=float)
    zmax = float(np.max(z))
    grid = np.linspace(0.0, zmax * 1.001 + 1e-6, 6000)
    ez = e_z(grid, omega_m=omega_m, w0=w0, wa=wa)
    if np.any(~np.isfinite(ez)):
        return np.full_like(z, np.nan, dtype=float)

    integral = cumulative_trapezoid(1.0 / ez, grid, initial=0.0)
    chi = np.interp(z, grid, integral)
    dl_mpc = (C_KM_S / h0) * (1.0 + z) * chi

    if np.any(dl_mpc <= 0.0) or not np.all(np.isfinite(dl_mpc)):
        return np.full_like(z, np.nan, dtype=float)

    return 5.0 * np.log10(dl_mpc) + 25.0


def gaussian_template(z: np.ndarray, c: float, s: float) -> np.ndarray:
    return np.exp(-0.5 * ((z - c) / s) ** 2)


def standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = x - np.mean(x)
    rms = np.sqrt(np.mean(y**2))
    if rms <= 0.0:
        raise ValueError("Cannot standardize zero-RMS template")
    return y / rms


def low_mid_template(z: np.ndarray) -> np.ndarray:
    """Primary low-z / mid-z nuisance template, standardized on Union3 nodes."""
    t = gaussian_template(z, 0.09, 0.06) - gaussian_template(z, 0.775, 0.10)
    return standardize(t)


def gls_fit(
    mu_obs: np.ndarray,
    precision: np.ndarray,
    mu_model: np.ndarray,
    template: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray]:
    """
    Generalized least squares fit for intercept M and optionally nuisance amplitude a.

    Model:
        mu_obs = mu_model + M + a*T(z)

    Returns:
        chi2_min, beta
    where beta = [M] or [M, a].
    """
    if np.any(~np.isfinite(mu_model)):
        return np.inf, np.array([np.nan])

    residual = mu_obs - mu_model

    if template is None:
        xmat = np.ones((len(mu_obs), 1))
    else:
        xmat = np.column_stack([np.ones(len(mu_obs)), template])

    fisher = xmat.T @ precision @ xmat
    rhs = xmat.T @ precision @ residual

    try:
        beta = np.linalg.solve(fisher, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(fisher) @ rhs

    final_residual = residual - xmat @ beta
    chi2 = float(final_residual.T @ precision @ final_residual)
    return chi2, beta


def fit_lcdm(
    z: np.ndarray,
    mu_obs: np.ndarray,
    precision: np.ndarray,
    template: Optional[np.ndarray] = None,
) -> FitResult:
    mu_model = distance_modulus(z, omega_m=OMEGA_M_FIXED, w0=-1.0, wa=0.0)
    chi2, beta = gls_fit(mu_obs, precision, mu_model, template)
    params = {"Omega_m": OMEGA_M_FIXED, "w0": -1.0, "wa": 0.0, "M": beta[0]}
    if template is not None:
        params["A_nuis"] = beta[1]
    return FitResult(
        "LCDM" if template is None else "LCDM+nuisance",
        chi2,
        params,
        0 if template is None else 1,
    )


def fit_cpl(
    z: np.ndarray,
    mu_obs: np.ndarray,
    precision: np.ndarray,
    template: Optional[np.ndarray] = None,
) -> FitResult:
    """Fit CPL parameters w0, wa at fixed Omega_m, with M and optional nuisance analytically marginalized."""

    def objective(x: np.ndarray) -> float:
        w0, wa = float(x[0]), float(x[1])
        mu_model = distance_modulus(z, omega_m=OMEGA_M_FIXED, w0=w0, wa=wa)
        chi2, _ = gls_fit(mu_obs, precision, mu_model, template)
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
        res = minimize(objective, start, method="Nelder-Mead", options={"maxiter": 5000})
        clipped = np.array([
            np.clip(res.x[0], bounds[0][0], bounds[0][1]),
            np.clip(res.x[1], bounds[1][0], bounds[1][1]),
        ])
        res2 = minimize(objective, clipped, method="L-BFGS-B", bounds=bounds, options={"maxiter": 5000})
        candidate = res2 if res2.fun <= res.fun else res
        if best is None or candidate.fun < best.fun:
            best = candidate

    w0, wa = float(best.x[0]), float(best.x[1])
    mu_model = distance_modulus(z, omega_m=OMEGA_M_FIXED, w0=w0, wa=wa)
    chi2, beta = gls_fit(mu_obs, precision, mu_model, template)

    params = {"Omega_m": OMEGA_M_FIXED, "w0": w0, "wa": wa, "M": beta[0]}
    if template is not None:
        params["A_nuis"] = beta[1]

    return FitResult("CPL" if template is None else "CPL+nuisance", chi2, params, 2 if template is None else 3)


def aic_gain(delta_chi2: float, added_parameters: int) -> float:
    """AIC gain relative to baseline: Delta chi2 - 2*k."""
    return float(delta_chi2 - 2.0 * added_parameters)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to mu_mat_union3_cosmo=2_mu.fits. Default searches data/ then repository root.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/union3_fixed_omega_summary.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    if args.data is None:
        candidates = [
            Path("data/mu_mat_union3_cosmo=2_mu.fits"),
            Path("mu_mat_union3_cosmo=2_mu.fits"),
        ]
        data_path = next((p for p in candidates if p.exists()), candidates[0])
    else:
        data_path = args.data

    z, mu_obs, precision = load_union3_compressed(data_path)
    template = low_mid_template(z)

    lcdm = fit_lcdm(z, mu_obs, precision, template=None)
    cpl = fit_cpl(z, mu_obs, precision, template=None)
    lcdm_nuis = fit_lcdm(z, mu_obs, precision, template=template)
    cpl_nuis = fit_cpl(z, mu_obs, precision, template=template)

    delta_cpl = lcdm.chi2 - cpl.chi2
    delta_nuis = lcdm.chi2 - lcdm_nuis.chi2
    delta_cpl_after_nuis = lcdm_nuis.chi2 - cpl_nuis.chi2

    rows = [
        {
            "model": lcdm.model,
            "chi2": lcdm.chi2,
            "delta_chi2_vs_lcdm": 0.0,
            "aic_gain_vs_lcdm": 0.0,
            **lcdm.params,
        },
        {
            "model": cpl.model,
            "chi2": cpl.chi2,
            "delta_chi2_vs_lcdm": delta_cpl,
            "aic_gain_vs_lcdm": aic_gain(delta_cpl, 2),
            **cpl.params,
        },
        {
            "model": lcdm_nuis.model,
            "chi2": lcdm_nuis.chi2,
            "delta_chi2_vs_lcdm": delta_nuis,
            "aic_gain_vs_lcdm": aic_gain(delta_nuis, 1),
            **lcdm_nuis.params,
        },
        {
            "model": cpl_nuis.model,
            "chi2": cpl_nuis.chi2,
            "delta_chi2_vs_lcdm": lcdm.chi2 - cpl_nuis.chi2,
            "aic_gain_vs_lcdm": aic_gain(lcdm.chi2 - cpl_nuis.chi2, 3),
            "delta_chi2_after_nuis": delta_cpl_after_nuis,
            "aic_gain_after_nuis": aic_gain(delta_cpl_after_nuis, 2),
            **cpl_nuis.params,
        },
    ]

    df = pd.DataFrame(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print("\nUnion3 fixed-Omega_m compressed-data diagnostic")
    print(f"Data file: {data_path}")
    print(f"N nodes: {len(z)}")
    print(f"Fixed Omega_m: {OMEGA_M_FIXED}")
    print("\nModel summary:")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nKey expected comparisons:")
    print(f"CPL Delta chi2 vs LCDM             = {delta_cpl:.6f}   expected approx 7.536")
    print(f"Low-mid nuisance Delta chi2 vs LCDM = {delta_nuis:.6f}   expected approx 17.120")
    print(f"CPL after nuisance Delta chi2       = {delta_cpl_after_nuis:.6f}   expected approx 0.870")
    print(f"\nSaved summary to: {args.out}")


if __name__ == "__main__":
    main()
