# Cobaya public-data cross-check

This folder is for Cobaya YAML files and notes related to the public DESI BAO + compressed CMB prior + Union3 cross-check.

This is not the official DESI/CMB/Union3 full likelihood.

## Purpose

The Cobaya cross-check tests whether the same low-z / mid-z nuisance direction also reduces the apparent CPL preference in a simplified public-data joint setup.

## Expected qualitative result

In the public-data proxy setup:

- CPL improvement without nuisance: Delta chi2 ≈ 8.29
- CPL improvement after nuisance: Delta chi2 ≈ 0.52

These numbers are diagnostic only and should not be treated as an official DESI result.

## Files to add later

Expected files may include:

- joint_lcdm.yaml
- joint_cpl.yaml
- joint_lcdm_nuisance.yaml
- joint_cpl_nuisance.yaml
- notes_public_joint_crosscheck.md

## Important limitation

This folder does not reproduce the official DESI likelihood. It only documents a simplified public-data cross-check used as a robustness diagnostic.
