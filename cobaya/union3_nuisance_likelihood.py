"""
Union3 compressed-data nuisance likelihood for Cobaya.

Purpose:
    Public-data cross-check for the Union3 low-z/mid-z nuisance direction.

This is not an official DESI collaboration likelihood.
It is a transparent public-data likelihood component for Union3 compressed distances.

Expected Union3 FITS structure, following rubind/union3_release README:
    - first row contains redshifts
    - first column contains observed distance moduli
    - remaining block contains inverse covariance matrix

Cobaya usage:
    likelihood:
      union3_nuisance_likelihood.Union3NuisanceLikelihood:
        data_file: mu_mat_union3_cosmo=2_mu.fits
        use_nuisance: true

Required sampled parameters:
    M_SN: additive SN magnitude offset
    A_nuis: nuisance amplitude in mag, only used if use_nuisance=true
"""

import numpy as np
from cobaya.likelihood import Likelihood

try:
    from astropy.io import fits
except Exception as exc:  # pragma: no cover
    fits = None
    _fits_import_error = exc


class Union3NuisanceLikelihood(Likelihood):
    data_file: str = "mu_mat_union3_cosmo=2_mu.fits"
    use_nuisance: bool = False
    z_low: float = 0.09
    sig_low: float = 0.06
    z_mid: float = 0.775
    sig_mid: float = 0.10

    def initialize(self):
        if fits is None:
            raise RuntimeError(f"astropy is required to read FITS files: {_fits_import_error}")

        raw = fits.getdata(self.data_file)
        mat = np.array(raw, dtype=float)

        # Robust parser for Union3 matrix layout.
        # README says first row = z, first column = mu, remaining matrix = inverse covariance.
        self.z = np.array(mat[0, 1:], dtype=float)
        self.mu_obs = np.array(mat[1:, 0], dtype=float)
        self.invcov = np.array(mat[1:, 1:], dtype=float)

        if len(self.z) != len(self.mu_obs):
            raise ValueError(
                f"Union3 FITS parse mismatch: len(z)={len(self.z)}, len(mu_obs)={len(self.mu_obs)}. "
                "Check matrix orientation."
            )
        if self.invcov.shape != (len(self.z), len(self.z)):
            raise ValueError(
                f"Inverse covariance shape {self.invcov.shape} does not match N={len(self.z)}."
            )

        t = self._raw_template(self.z)
        # Standardize to mean 0 and standard deviation 1 for stable amplitude interpretation.
        self.template = (t - np.mean(t)) / np.std(t)

    def get_requirements(self):
        # Cobaya provider returns angular diameter distances in Mpc for these redshifts.
        return {"angular_diameter_distance": {"z": self.z}}

    def _raw_template(self, z):
        g_low = np.exp(-0.5 * ((z - self.z_low) / self.sig_low) ** 2)
        g_mid = np.exp(-0.5 * ((z - self.z_mid) / self.sig_mid) ** 2)
        return g_low - g_mid

    def logp(self, M_SN=0.0, A_nuis=0.0, **params_values):
        da = np.array(self.provider.get_angular_diameter_distance(self.z), dtype=float)
        dl = da * (1.0 + self.z) ** 2
        mu_th = 5.0 * np.log10(dl) + 25.0

        model = mu_th + M_SN
        if self.use_nuisance:
            model = model + A_nuis * self.template

        resid = self.mu_obs - model
        chi2 = float(resid @ self.invcov @ resid)
        return -0.5 * chi2
