# Union3 low-z / mid-z nuisance robustness diagnostic

This repository reproduces a compressed-data robustness diagnostic for the public Union3 supernova release.

It tests whether a low-z / mid-z redshift-dependent nuisance direction can absorb part of the CPL w0-wa preference in the public Union3 compressed data.

This is not an official DESI/CMB/Union3 full likelihood analysis. It is a public-data diagnostic and falsification-test proposal.

## Main expected results

- Union3 fixed-Omega_m CPL improvement: Delta chi2 ≈ 7.536
- Low-z / mid-z Gaussian nuisance improvement: Delta chi2 ≈ 17.120
- CPL after nuisance: Delta chi2 ≈ 0.870
- Public joint cross-check: CPL improvement drops from ≈ 8.29 to ≈ 0.52 after nuisance
- Random-template null test: 0/200 random templates exceeded the original template; the closest random template was near the original, so this is a ranking/null diagnostic rather than proof of template uniqueness.
- Note: The repository scripts use the current fixed-Omega_m compressed-data protocol. Some earlier draft/preprint robustness tables used a simplified diagnostic setup, so absolute Delta chi2 values may differ from the current repository outputs.

## Data

This project uses public Union3 and Union3.1 data products from the rubind/union3_release repository.

The original data files are not redistributed here unless licensing and file-size checks are completed. Users should download the public data directly from the Union3 release repository.

## Status

Private work-in-progress reproduction package.

## Limitations

This repository does not reproduce the official DESI/CMB/Union3 full likelihood. It reproduces compressed-data and public-data diagnostic tests only.

## How to run

Install dependencies:

```bash
pip install -r requirements.txt
```
Download the public Union3 compressed FITS file from the `rubind/union3_release` repository and place it here:

```text
data/mu_mat_union3_cosmo=2_mu.fits
```

For the three-version comparison, also place the Union3.1 compressed files here:

```text
data/mu_mat_union3.1_UNITY1.7_template_cosmo=2_0_mu.fits
data/mu_mat_union3.1_UNITY1.8_template_cosmo=2_0_mu.fits
```

Run the main diagnostics:

```bash
python scripts/01_union3_fixed_omega_test.py
python scripts/02_template_robustness_grid.py
python scripts/03_random_template_null.py
python scripts/04_leave_one_region_out.py
python scripts/05_three_version_comparison.py
python scripts/06_cumulative_lowz_removal.py
```
The scripts write output CSV files into the `results/` folder.
