import numpy as np
from cobaya.likelihoods.sn.union3 import Union3


class Union3Nuisance(Union3):
    """Union3 likelihood with a redshift-dependent nuisance template.

        mu_eff(z) = mu_theory(z) + a_nuis * T(z)

    Implemented by shifting the data vector mag -> mag - a_nuis * T(z),
    which is mathematically equivalent (same chi2 residual).
    T(z) = G(z; 0.09, 0.06) - G(z; 0.775, 0.10), zero mean, unit RMS.
    """

    # Ütle Cobayale, et see likelihood ootab parameetrit a_nuis
    params = {"a_nuis": None}

    def configure(self):
        super().configure()
        z = np.asarray(self.zcmb, dtype=float)
        T = (np.exp(-0.5 * ((z - 0.09) / 0.06) ** 2)
             - np.exp(-0.5 * ((z - 0.775) / 0.10) ** 2))
        T = T - T.mean()
        T = T / np.sqrt((T ** 2).mean())
        self._template = T

    def logp(self, **params):
        a = params.get("a_nuis", 0.0)
        saved = self.mag.copy()
        try:
            self.mag = saved - a * self._template
            out = super().logp(**params)
        finally:
            self.mag = saved
        return out
