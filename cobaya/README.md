# Joint cross-checks (Cobaya): Tables 2 and 3

This folder contains the Cobaya configuration and analysis scripts for the two joint
cross-checks in the note:

- **Table 3 — full-likelihood cross-check:** Planck NPIPE CamSpec TTTEEE + low-l TT/EE +
  DESI DR1 BAO + Union3, with the low-z/mid-z nuisance amplitude `a_nuis` sampled directly
  inside the Union3 likelihood.
- **Table 2 — compressed-prior proxy:** DESI DR1 BAO + Union3 with a compressed CMB prior
  (lighter, faster public cross-check).

These runs are **not executed by this repository** — they require external likelihood codes
and data (Planck, DESI, Union3) and produce multi-GB MCMC chains. This folder provides the
exact configuration so the runs can be reproduced on a machine with those dependencies
installed. The resulting numbers are reported in the note; the chains themselves are not
committed (see `.gitignore`).

## What you must install separately

The configs assume a working [Cobaya](https://cobaya.readthedocs.io) + CAMB installation
and the following likelihoods/data, none of which are redistributed here:

- `planck_2018_lowl.TT`, `planck_2018_lowl.EE` (Planck 2018 low-l)
- `planck_NPIPE_highl_CamSpec.TTTEEE` (Planck NPIPE CamSpec; dataset `CamSpec_NPIPE_12_6`)
- `bao.desi_2024_bao_all` (DESI DR1 / 2024 BAO)
- `sn.union3` (Union3 supernova likelihood, ships with the DESI Cobaya likelihoods)

Install the codes and download the data with, e.g.:

```bash
pip install cobaya
cobaya-install fullplanck_cpl_nuis.yaml --packages-path /path/to/cobaya_packages
```

`cobaya-install` will fetch CAMB and the Planck/DESI/Union3 likelihood data. Point every
run at the same `--packages-path`. The analysis scripts (`build_table2.py`,
`fullplanck_w0wa_aic.py`, `plot_contours.py`) use `getdist`, which is installed as a
dependency of Cobaya.

### Software versions used for the published numbers

- Cobaya 3.6.2, CAMB 1.6.7 (full-likelihood runs; `dark_energy_model: ppf`)
- Planck NPIPE CamSpec dataset `CamSpec_NPIPE_12_6_cl.dataset`
  (`use_cl: 143x143 217x217 143x217 TE EE`)
- The compressed-prior proxy (Table 2) was run with Cobaya 3.4.1 / CAMB 1.5.4.

## The nuisance likelihood: `my_union3.py`

`Union3Nuisance` subclasses Cobaya's built-in `sn.union3` and adds one sampled parameter
`a_nuis`. It applies the same redshift template used throughout the note,

    T(z) = G(z; 0.09, 0.06) - G(z; 0.775, 0.10),   zero mean, unit RMS,

by shifting the data vector `mag -> mag - a_nuis * T(z)` (mathematically equivalent to
`mu_theory -> mu_theory + a_nuis * T(z)`; identical chi2 residual). This is the same
template and convention as the compressed-data scripts in `../scripts/`.

The configs reference it as `my_union3.Union3Nuisance` with `python_path: .`, so **run the
configs from inside this `cobaya/` folder** (or adjust `python_path`) so that
`my_union3.py` is importable.

## Files

Full-likelihood (Table 3) — four models:

| File | Model | Output chain |
|------|-------|--------------|
| `fullplanck_lcdm.yaml`       | LCDM                | `chains/fullplanck_lcdm` |
| `fullplanck_lcdm_nuis.yaml`  | LCDM + nuisance     | `chains/fullplanck_lcdm_nuis` |
| `fullplanck_cpl.yaml`        | CPL                 | `chains/fullplanck_cpl` |
| `fullplanck_cpl_nuis.yaml`   | CPL + nuisance      | `chains/fullplanck` |

Compressed-prior proxy (Table 2):

| File | Model | Output chain |
|------|-------|--------------|
| `proxy_base_w_wa.yaml`     | CPL (DESI BAO + Union3) | `chains/baseline` |
| `proxy_nuisance_w_wa.yaml` | CPL + nuisance          | `chains/nuisance` |

Analysis / helper scripts:

- `build_table2.py` — reads the four full-likelihood chains and prints Table 3
  (best-fit chi2, Delta chi2, AIC gain, and the 2D distance of the w0-wa mean from LCDM).
- `fullplanck_w0wa_aic.py` — CPL vs CPL+nuisance comparison plus the w0-wa contour figure.
- `plot_contours.py` — w0-wa posterior contour plot from the CPL+nuisance chain.
- `make_lcdm_yaml.py` — helper to derive an LCDM config (w0=-1, wa=0 fixed) from a CPL config.

## How to run

From inside this folder, after `cobaya-install`:

```bash
# Full-likelihood: run all four models (each is a multi-chain MCMC; takes a while)
cobaya-run fullplanck_lcdm.yaml      --packages-path /path/to/cobaya_packages
cobaya-run fullplanck_lcdm_nuis.yaml --packages-path /path/to/cobaya_packages
cobaya-run fullplanck_cpl.yaml       --packages-path /path/to/cobaya_packages
cobaya-run fullplanck_cpl_nuis.yaml  --packages-path /path/to/cobaya_packages

# Build Table 3 from the chains
python build_table2.py

# CPL vs CPL+nuisance comparison + w0-wa figure
python fullplanck_w0wa_aic.py
```

For the compressed-prior proxy (Table 2), run `proxy_base_w_wa.yaml` and
`proxy_nuisance_w_wa.yaml` the same way.

## Caveats (read before quoting numbers)

- **Chains are not committed.** They are large (GB-scale) and excluded by `.gitignore`.
  You regenerate them with the commands above.
- **Best-sample chi2.** `build_table2.py` and `fullplanck_w0wa_aic.py` use the minimum chi2
  over the MCMC samples as a proxy for the global best fit. For final publication numbers,
  run `cobaya-run <input>.yaml --minimize` per model. The Gelman-Rubin convergence for the
  full-likelihood runs reached R-1 = 0.047 on the means (R-1 ~ 0.15 on the bounds), so these
  are public cross-checks, not collaboration-grade parameter estimation.
- **Official DESI config not redistributed.** The official DESI+CMB+Union3 setup uses the
  DESI internal Cobaya bindings and NERSC-hosted data paths, which are not included here.
  The proxy configs above are a public, runnable approximation of Table 2.
- These are public reimplementations, not the official, internally validated
  DESI+CMB+Union3 likelihood with a collaboration-modelled systematic.
