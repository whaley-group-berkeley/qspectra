"""Defines Hamiltonian class to use for dynamics"""
from functools import wraps
import inspect
import numpy as np
import scipy.linalg
import scipy.sparse

from constants import CM_FS, GAUSSIAN_SD_FWHM
from operator_tools import (operator_extend, S_commutator, S_left, to_basis,
                            transition_operator, unit_vec, BasisError)
from utils import imemoize


def transform_out_basis(default='site', from_basis='site'):
    """Function that returns a decorator to add optional basis or sparifying
    transformations to a function that returns states or operators
    """
    def decorator(func):
        @wraps(func)
        def wrapper(hamiltonian, *args, **kwargs):
            basis = kwargs.pop('basis', default)
            try:
                subspace = kwargs['subspace']
            except KeyError:
                arg_names = inspect.getargspec(func).args
                subspace = args[arg_names.index('subspace') - 1]

            X = func(hamiltonian, *args, **kwargs)
            if basis != from_basis:
                if basis == 'exciton' and from_basis == 'site':
                    X = hamiltonian.system.site_to_exc(X, subspace)
                elif basis == 'site' and from_basis == 'exciton':
                    X = hamiltonian.system.exc_to_site(X, subspace)
                else:
                    raise BasisError(('invalid basis transform {1!r} -> '
                                      '{0!r}').format(basis, from_basis))
            return X
        return wrapper
    return decorator


COORD_MAP = {'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}


class Hamiltonian(object):
    def __init__(self, system, bath, dipoles=None):
        self.system = system
        self.bath = bath
        self.dipoles = dipoles

    def child(self, system=None, bath=None, dipoles=None):
        return type(self)(system or self.system,
                          bath or self.bath,
                          dipoles or self.dipoles)

    def to_rotating_frame(self, *args, **kwargs):
        return self.child(system=self.system.to_rotating_frame(*args, **kwargs))

    def sample(self, N):
        for sys in self.system.sample(N):
            yield self.child(system=sys)

    @property
    def freq_step(self):
        return self.system.freq_step

    @property
    def time_step(self):
        return self.system.time_step

    # system-field coupling methods:
    @transform_out_basis(from_basis='site')
    def dipole_operator(self, polarization, subspace='ge'):
        try:
            polarization = COORD_MAP[polarization]
        except (TypeError, KeyError):
            pass
        N = self.system.n_sites
        trans_ops = [transition_operator(i, N, subspace) for i in xrange(N)]
        return np.einsum('nij,nk,k->ij', trans_ops, self.dipoles, polarization)

    @imemoize
    def dipole_destroy_evolve(self, *args, **kwargs):
        X = np.triu(self.dipole_operator(*args, **kwargs))
        return -1j * S_commutator(X)

    @imemoize
    def dipole_destroy_left(self, *args, **kwargs):
        X = np.triu(self.dipole_operator(*args, **kwargs))
        return S_left(X)

    # system-bath coupling methods:
    def to_temperature(self, temperature):
        return self.child(bath=self.bath.to_temperature(temperature))

    @property
    def thermal_state(self):
        return self.system.thermal_state(self.bath.temperature)

    @imemoize
    def system_bath_coupling(self, subspace=None):
        N = self.system.n_sites
        return [operator_extend(np.diag(unit_vec(n, N)), subspace)
                for n in xrange(N)]

    def __repr__(self):
        return "{0}(system={1}, bath={2}, dipoles={3})".format(
            type(self).__name__, self.system, self.bath, self.dipoles)


class ElectronicHamiltonian(object):
    def __init__(self, H_1, K_2=None, disorder_fwhm=None, ref_system=None):
        self.H_1 = H_1
        self.K_2 = K_2
        self.disorder_fwhm = disorder_fwhm
        self.ref_system = ref_system if ref_system is not None else self

    @property
    def n_sites(self):
        return len(self.H_1)

    def n_states(self, subspace):
        return len(self.H(subspace))

    FREQ_EXTRA = 100.

    @property
    def freq_step(self):
        """Calculate an appropriate sampling rate in 1/fs according to the
        Nyquist theorem so that all frequencies of the provided Hamiltonian can
        be resolved"""
        energies = self.ref_system.E('012')
        freq_span = energies.max() - energies.min()
        return 2 * CM_FS * (freq_span + self.FREQ_EXTRA)

    @property
    def time_step(self):
        return 1.0 / self.freq_step

    @imemoize
    def eig(self, subspace):
        E, U = scipy.linalg.eigh(self.H(subspace))
        return (E, U)

    def E(self, subspace):
        return self.eig(subspace)[0]

    def U(self, subspace):
        return self.eig(subspace)[1]

    def site_to_exc(self, X, subspace=None):
        return to_basis(self.U(subspace), X)

    def exc_to_site(self, X, subspace=None):
        return to_basis(self.U(subspace).T.conj(), X)

    @imemoize
    def H(self, subspace):
        return operator_extend(self.H_1, subspace, self.K_2)

    @imemoize
    def to_rotating_frame(self, rw_freq=None):
        if rw_freq is None:
            rw_freq = self.central_freq
        H_1_prime = self.H_1 - rw_freq * np.identity(len(self.H_1))
        new_ref_system = (self.ref_system.to_rotating_frame(rw_freq)
                          if (self.ref_system is not self) else None)
        return type(self)(H_1_prime, self.K_2, self.disorder_fwhm,
                          ref_system=new_ref_system)

    @property
    def central_freq(self):
        return np.mean(self.ref_system.E('e'))

    def sample(self, ensemble_size=1):
        for _ in xrange(ensemble_size):
            H_1_prime = (self.H_1 + self.disorder_fwhm * GAUSSIAN_SD_FWHM
                         * np.diag(np.random.randn(self.n_sites)))
            yield type(self)(H_1_prime, self.K_2, ref_system=self)

    def thermal_state(self, temperature):
        diag_scale = self.central_freq * np.identity(self.n_sites)
        rho = scipy.linalg.expm(-(self.H_1 - diag_scale) / temperature)
        return rho / np.trace(rho)

    def __repr__(self):
        return ("{0}(H_1={1}, K_2={2}, disorder_fwhm={3}, ref_system={4})"
                ).format(type(self).__name__, self.H_1, self.K_2,
                         self.disorder_fwhm,
                         None if self.ref_system is self else self.ref_system)
