#!/usr/bin/env python
"""
02_template_robustness_grid.py

Template robustness grid for the Union3 compressed-data low-z / mid-z nuisance
diagnostic.

This script scans nearby two-Gaussian nuisance templates around the primary
template:

    T(z) = G(z; 0.09, 0.06) - G(z; 0.775, 0.10)

and checks whether the nuisance improvement remains positive when the template
centers and widths are varied.

Expected qualitative result:
    - 81 template variants
    - nuisance Delta chi2 remains positive for all variants
    - output saved to results/template_robustness_grid.csv
      and results/template_robustness_grid_summary.csv

Required input file:
    data/mu_mat_union3_cosmo=2_mu.fits

Run from repository root:
    python scripts/02_template_robustness_grid.py
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


def make_template(z: np.ndarray, c_low: float, s_low: float, c_mid: float, s_mid: float) -> np.ndarray:
    return standardize(gaussian(z, c_low, s_low) - gaussian(z, c_mid, s_mid))


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


def fit_cpl_chi2(
    z: np.ndarray,
    mu_obs: np.ndarray,
    precision: np.ndarray,
    template: Optional[np.ndarray] = None,
) -> float:
    """
    Fit CPL chi2 at fixed Omega_m.

    Nelder-Mead is used only to find a good starting point.
    The accepted final result is always taken from a bounded L-BFGS-B run,
    so w0 and wa remain inside the declared parameter bounds.

    Some SciPy/Windows runs can report success=False even when a finite,
    useful bounded minimum is returned. For this diagnostic grid we accept
    the best finite bounded result and only fail if no finite bounded result
    is found.
    """

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

        if np.isfinite(res_bounded.fun) and float(res_bounded.fun) < best_fun:
            best_fun = float(res_bounded.fun)

    if not np.isfinite(best_fun):
        raise RuntimeError("CPL fit failed: no finite bounded optimizer result was found.")

    return best_fun


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/mu_mat_union3_cosmo=2_mu.fits"))
    parser.add_argument("--outdir", type=Path, default=Path("results"))
    args = parser.parse_args()

    z, mu_obs, precision = load_union3_compressed(args.data)
    args.outdir.mkdir(parents=True, exist_ok=True)

    lcdm_chi2, _ = fit_lcdm_chi2(z, mu_obs, precision, template=None)
    cpl_chi2 = fit_cpl_chi2(z, mu_obs, precision, template=None)
    cpl_delta = lcdm_chi2 - cpl_chi2

    # 3 x 3 x 3 x 3 = 81 variants around the primary template.
    # We vary low-z center, mid-z center, low-z width, and mid-z width.
    low_centers = [0.07, 0.09, 0.11]
    mid_centers = [0.70, 0.775, 0.85]
    low_widths = [0.04, 0.06, 0.08]
    mid_widths = [0.08, 0.10, 0.12]

    rows = []
    for c_low in low_centers:
        for c_mid in mid_centers:
            for s_low in low_widths:
                for s_mid in mid_widths:
                    template = make_template(z, c_low, s_low, c_mid, s_mid)
                    nuisance_chi2, beta = fit_lcdm_chi2(z, mu_obs, precision, template=template)
                    cpl_nuis_chi2 = fit_cpl_chi2(z, mu_obs, precision, template=template)

                    rows.append(
                        {
                            "c_low": c_low,
                            "s_low": s_low,
                            "c_mid": c_mid,
                            "s_mid": s_mid,
                            "lcdm_chi2": lcdm_chi2,
                            "cpl_chi2": cpl_chi2,
                            "cpl_delta_chi2": cpl_delta,
                            "nuisance_chi2": nuisance_chi2,
                            "nuisance_delta_chi2": lcdm_chi2 - nuisance_chi2,
                            "nuisance_amplitude": beta[1],
                            "cpl_nuisance_chi2": cpl_nuis_chi2,
                            "cpl_after_nuisance_delta_chi2": nuisance_chi2 - cpl_nuis_chi2,
                        }
                    )

    df = pd.DataFrame(rows)
    grid_path = args.outdir / "template_robustness_grid.csv"
    df.to_csv(grid_path, index=False)

    summary = pd.DataFrame(
        [
            {
                "number_of_templates": len(df),
                "lcdm_chi2": lcdm_chi2,
                "cpl_chi2": cpl_chi2,
                "cpl_delta_chi2_vs_lcdm": cpl_delta,
                "nuisance_delta_chi2_median": df["nuisance_delta_chi2"].median(),
                "nuisance_delta_chi2_min": df["nuisance_delta_chi2"].min(),
                "nuisance_delta_chi2_max": df["nuisance_delta_chi2"].max(),
                "all_nuisance_positive": bool((df["nuisance_delta_chi2"] > 0).all()),
                "median_cpl_after_nuisance_delta_chi2": df["cpl_after_nuisance_delta_chi2"].median(),
            }
        ]
    )

    summary_path = args.outdir / "template_robustness_grid_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nTemplate robustness grid")
    print(f"Data file: {args.data}")
    print(f"Number of templates: {len(df)}")
    print("\nSummary:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nSaved grid to: {grid_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()