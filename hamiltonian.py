from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np
import scipy.linalg

from constants import GAUSSIAN_SD_FWHM
from operator_tools import transition_operator, operator_extend, unit_vec
from utils import imemoize


class HamiltonianError(Exception):
    """
    Error class for Hamiltonian errors
    """


class Hamiltonian(object):
    """
    Parent class for Hamiltonian objects
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def H(self, subspace):
        """
        Returns the system Hamiltonian in the given Hilbert subspace as a matrix
        """

    @abstractmethod
    def ground_state(self, subspace):
        """
        Returns the ground electronic state of this Hamiltonian as a density
        operator
        """

    @abstractmethod
    def in_rotating_frame(self, rw_freq=None):
        """
        Returns a new Hamiltonian shifted to the rotating frame at the given
        frequency

        By default, sets the rotating frame to the central frequency.
        """

    @abstractmethod
    def dipole_operator(self, subspace='gef', polarization='x',
                        transitions='-+'):
        """
        Return the matrix representation in the given subspace of the requested
        dipole operator
        """

    @abstractmethod
    def system_bath_couplings(self, subspace='gef'):
        """
        Return a list of matrix representations in the given subspace of the
        system-bath coupling operators
        """

    def n_states(self, subspace):
        return len(self.H(subspace))

    @imemoize
    def eig(self, subspace):
        """
        Returns the eigensystem solution E, U for this Hamiltonian in the given
        subspace
        """
        E, U = scipy.linalg.eigh(self.H(subspace))
        return (E, U)

    def E(self, subspace):
        """
        Returns the eigen-energies of this Hamiltonian in the given subspace
        """
        return self.eig(subspace)[0]

    def U(self, subspace):
        """
        Returns the matrix which diagonalizes this Hamiltonian in the given
        subspace
        """
        return self.eig(subspace)[1]

    @property
    def mean_excitation_freq(self):
        """
        Average excited state transition energy
        """
        return np.mean(self.E('e')) + self.energy_offset

    @property
    def freq_step(self):
        """
        An appropriate sampling rate, according to the Nyquist theorem, so that
        all frequencies of the Hamiltonian can be resolved
        """
        freq_span = self.E('gef').max() - self.E('gef').min()
        return 2 * (freq_span + self.energy_spread_extra)

    @property
    def time_step(self):
        return 1.0 / self.freq_step


COORD_MAP = {'x': np.array([1, 0, 0]),
             'y': np.array([0, 1, 0]),
             'z': np.array([0, 0, 1])}


def polarization_vector(p):
    """
    Cast a polarization vector given by a list of three points or 'x', 'y' or
    'z' into a 3D vector
    """
    try:
        return COORD_MAP[p]
    except (TypeError, KeyError):
        return np.asanyarray(p)


class ElectronicHamiltonian(Hamiltonian):
    """
    Hamiltonian for an electronic system with coupling to an external field
    and an identical bath at each pigment

    Properties
    ----------
    H_1exc : np.ndarray
        Matrix representation of this hamiltonian in the 1-excitation subspace
    energy_offset : number, optional
        Constant energy offset of the diagonal entries in H_1exc from the ground
        state energy.
    bath : bath.Bath, optional
        Object containing the bath information (i.e., correlation function and
        temperature). Each site is assumed to be linearly coupled to an
        identical bath of this form.
    dipoles : np.ndarray, optional
        n x 3 array of dipole moments for each site.
    energy_spread_extra : float, optional (default 100)
        Default extra frequency to add to the spread of energies when
        determining the frequency step size automatically.
    """
    def __init__(self, H_1exc, energy_offset=0, bath=None, dipoles=None,
                 energy_spread_extra=100.0):
        self.H_1exc = H_1exc
        self.energy_offset = energy_offset
        self.bath = bath
        self.dipoles = dipoles
        self.energy_spread_extra = energy_spread_extra

    @property
    def n_sites(self):
        return len(self.H_1exc)

    @imemoize
    def H(self, subspace):
        """
        Returns the system Hamiltonian in the given Hilbert subspace as a matrix
        """
        return operator_extend(self.H_1exc, subspace)

    def ground_state(self, subspace):
        """
        Returns the ground electronic state of this Hamiltonian as a density
        operator
        """
        N = self.n_states(subspace)
        state = np.zeros((N, N), dtype=complex)
        if 'g' in subspace:
            state[0, 0] = 1.0
        return state

    @imemoize
    def in_rotating_frame(self, rw_freq=None):
        """
        Returns a new Hamiltonian shifted to the rotating frame at the given
        frequency

        By default, sets the rotating frame to the central frequency.
        """
        if rw_freq is None:
            rw_freq = self.mean_excitation_freq
        shift = rw_freq - self.energy_offset
        H_1exc = self.H_1exc - shift * np.identity(len(self.H_1exc))
        return type(self)(H_1exc, rw_freq, self.bath, self.dipoles,
                          self.energy_spread_extra)

    def dipole_operator(self, subspace='gef', polarization='x',
                        transitions='-+'):
        """
        Return the matrix representation in the given subspace of the requested
        dipole operator
        """
        if self.dipoles is None:
            raise HamiltonianError('transition dipole moments undefined')
        trans_ops = [transition_operator(n, self.n_sites, subspace, transitions)
                     for n in xrange(self.n_sites)]
        return np.einsum('nij,nk,k->ij', trans_ops, self.dipoles,
                         polarization_vector(polarization))

    def system_bath_couplings(self, subspace='gef'):
        """
        Return a list of matrix representations in the given subspace of the
        system-bath coupling operators
        """
        if self.bath is None:
            raise HamiltonianError('bath undefined')
        return [operator_extend(np.diag(unit_vec(n, self.n_sites)), subspace)
                for n in xrange(self.n_sites)]
