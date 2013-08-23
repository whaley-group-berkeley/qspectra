"""Defines Bath classes to use in Hamiltonians"""
import numpy as np
from numpy import pi, tan, exp


class Bath(object):
    """
    Attributes
    ----------
    temperature : float
        Temperature of the bath.
    spectral_density : SpectralDensity
        Spectral density of the bath.
    """
    def corr_func_real(self, x):
        """
        Correlation function
        """
        T = self.temperature
        J = self.spectral_density_func
        J0 = self.spectral_density_limit_at_zero

        def n(x):
            return 1 / (exp(x / T) - 1)

        def J_anti(x):
            return J(x) if x >= 0 else -J(-x)

        if x == 0:
            return T * J0
        else:
            return (n(x) + 1) * J_anti(x)

    def spectral_density_func(self, x):
        """
        Functional form of the spectral density
        """
        raise NotImplementedError

    @property
    def spectral_density_limit_at_zero(self):
        """
        Value of spectral_density_func divided by x at 0
        """
        raise NotImplementedError


class ArbitraryBath(Bath):
    def __init__(self, temperature, spectral_density_func,
                 spectral_density_limit_at_zero):
        self.temperature = temperature
        self.spectral_density_func = spectral_density_func
        self.spectral_density_limit_at_zero = spectral_density_limit_at_zero


class UncoupledBath(Bath):
    def corr_func_complex(self, _):
        return 0 + 0j

    def spectral_density_func(self, _):
        return 0

    @property
    def spectral_density_limit_at_zero(self):
        return 0


class DebyeBath(Bath):
    def __init__(self, temperature, reorg_energy, cutoff_freq):
        self.temperature = temperature
        self.reorg_energy = reorg_energy
        self.cutoff_freq = cutoff_freq

    def spectral_density_func(self, x):
        return (2 * self.reorg_energy * self.cutoff_freq * x
                / (self.cutoff_freq ** 2 + x ** 2))

    @property
    def spectral_density_limit_at_zero(self):
        return 2 * self.reorg_energy / self.cutoff_freq

    def corr_func_complex(self, x, matsubara_cutoff=1000):
        """
        Full one-sided correlation function for Debye spectral density

        References
        ----------
        J. Chem. Phys. 112, 7953
        """
        T = self.temperature
        lmbda = self.reorg_energy
        gamma = self.cutoff_freq
        nu = 2 * pi * np.arange(matsubara_cutoff) * T
        if x == 0:
            return lmbda * (2 * T / gamma - 1j)
        else:
            return (lmbda * gamma
                    * ((1 / tan(gamma / (2 * T)) - 1j) / (gamma - 1j * x) +
                       4 * T * np.sum(nu / ((nu ** 2 - gamma ** 2)
                                            * (nu - 1j * x)))))
