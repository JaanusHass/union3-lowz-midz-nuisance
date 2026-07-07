from cobaya.likelihood import Likelihood

class CMBOmh2Prior(Likelihood):
    """
    Simple compressed CMB prior on physical matter density:
        omega_m h^2 = omega_b h^2 + omega_c h^2 + omega_nu h^2

    This is NOT a full Planck likelihood. It is a lightweight compressed prior
    for a public-data joint cross-check.
    """
    mean = 0.1430
    sigma = 0.0020
    omega_nu_h2 = 0.00064

    def logp(self, ombh2=None, omch2=None, **kwargs):
        omega_m_h2 = float(ombh2) + float(omch2) + float(self.omega_nu_h2)
        return -0.5 * ((omega_m_h2 - float(self.mean)) / float(self.sigma)) ** 2
