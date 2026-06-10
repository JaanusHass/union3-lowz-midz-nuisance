# Scripts

Python scripts for reproducing the Union3 compressed-data robustness diagnostics.

## Available scripts

- `01_union3_fixed_omega_test.py` - core fixed-Omega_m Union3 diagnostic
- `02_template_robustness_grid.py` - nearby low-z / mid-z template robustness grid
- `03_random_template_null.py` - random smooth-template null test
- `04_leave_one_region_out.py` - leave-one-redshift-region-out robustness test
- `05_three_version_comparison.py` - Union3 to Union3.1 compressed-data comparison
- `06_cumulative_lowz_removal.py` - cumulative low-z node-removal diagnostic
- `07_template_cpl_overlap.py` - template-CPL overlap (degeneracy) test and low-z / mid-z
  component decomposition (Sections 7.5-7.6); writes `results/template_cpl_overlap.csv`
  and `results/component_decomposition.csv`, and regenerates `paper/overlap_angle.{pdf,png}`
- `08_orthogonalized_null.py` - orthogonalized (beyond-CPL) null test: the improvement a
  template adds *after* CPL is in the model, original vs 200 random templates (same recipe
  and draw order as script 03); writes `results/orthogonalized_null.csv`

Run these scripts from the repository root, not from inside this folder.
