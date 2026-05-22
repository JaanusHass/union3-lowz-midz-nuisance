# Union3 low-z / mid-z nuisance robustness diagnostic

This repository reproduces a compressed-data robustness diagnostic for the public Union3 supernova release.

It tests whether a low-z / mid-z redshift-dependent nuisance direction can absorb part of the CPL w0-wa preference in the public Union3 compressed data.

This is not an official DESI/CMB/Union3 full likelihood analysis. It is a public-data diagnostic and falsification-test proposal.

## Main expected results

- Union3 fixed-Omega_m CPL improvement: Delta chi2 ≈ 7.536
- Low-z / mid-z Gaussian nuisance improvement: Delta chi2 ≈ 17.120
- CPL after nuisance: Delta chi2 ≈ 0.870
- Public joint cross-check: CPL improvement drops from ≈ 8.29 to ≈ 0.52 after nuisance
- Random-template null test: 0/200 random templates exceeded the original template

## Data

This project uses public Union3 and Union3.1 data products from the rubind/union3_release repository.

The original data files are not redistributed here unless licensing and file-size checks are completed. Users should download the public data directly from the Union3 release repository.

## Status

Private work-in-progress reproduction package.

## Limitations

This repository does not reproduce the official DESI/CMB/Union3 full likelihood. It reproduces compressed-data and public-data diagnostic tests only.
