"""Defines Bath classes to use in Hamiltonians"""
from collections import namedtuple

import numpy as np
from numpy import pi, tan, exp
from utils import Registry


bath_registry = Registry()


class Bath(object):
    def corr_func_real(self, *args, **kwargs):
        return self.corr_func_complex(*args, **kwargs).real


SpectralDensity = namedtuple('SpectralDensity', 'f, f0')


class UncoupledBath(Bath):
    def corr_func_complex(self, x):
        return 0 + 0j


class ArbitraryBath(Bath):
    def __init__(self, temperature, spectral_density):
        self.temperature = temperature
        self.spectral_density = spectral_density

    def to_temperature(self, temperature):
        return type(self)(temperature, self.spectral_density)

    def corr_func_real(self, x):
        T = self.temperature
        J = self.spectral_density.f
        J0 = self.spectral_density.f0

        def n(x):
            return 1 / (exp(x / T) - 1)

        def J_anti(x):
            return J(x) if x >= 0 else -J(-x)

        if x == 0:
            return T * J0
        else:
            return (n(x) + 1) * J_anti(x)

    def __repr__(self):
        return "{0}(temperature={1}, spectral_density={2})".format(
            type(self).__name__, self.temperature, self.spectral_density)


@bath_registry
def bath_ohmic_exp(temperature, reorg_energy, cutoff_freq):
    f = lambda x: reorg_energy * x / cutoff_freq * exp(-x / cutoff_freq)
    f0 = reorg_energy / cutoff_freq
    return ArbitraryBath(temperature, SpectralDensity(f, f0))


@bath_registry
class DebyeBath(Bath):
    def __init__(self, temperature, reorg_energy, cutoff_freq):
        self.temperature = temperature
        self.reorg_energy = reorg_energy
        self.cutoff_freq = cutoff_freq

    def to_temperature(self, temperature):
        return type(self)(temperature, self.reorg_energy, self.cutoff_freq)

    def corr_func_complex(self, x, matsubara_cutoff=1000):
        """Full one-sided correlation function for Debye spectral density

        References
        ----------
        J. Chem. Phys. 112, 7953
        """
        T, lmbda, gamma = self.temperature, self.reorg_energy, self.cutoff_freq
        nu = 2 * pi * np.arange(matsubara_cutoff) * T
        if x == 0:
            return lmbda * (2 * T / gamma - 1j)
        else:
            return (lmbda * gamma
                    * ((1 / tan(gamma / (2 * T)) - 1j) / (gamma - 1j * x) +
                       4 * T * np.sum(nu / ((nu ** 2 - gamma ** 2)
                                            * (nu - 1j * x)))))

    def __repr__(self):
        return "{0}(temperature={1}, reorg_energy={2}, cutoff_freq={3})".format(
            type(self).__name__, self.temperature, self.reorg_energy,
            self.cutoff_freq)
