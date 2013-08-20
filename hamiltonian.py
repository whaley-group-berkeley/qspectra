import numpy as np
import scipy.linalg

from constants import GAUSSIAN_SD_FWHM
from operator_tools import transition_operator, operator_extend, unit_vec
from utils import imemoize


COORD_MAP = {'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}


def polarization_vector(p):
    """
    Cast a polarization vector given by a list of three points or 'x', 'y' or
    'z' into a 3D vector
    """
    try:
        return COORD_MAP[p]
    except (TypeError, KeyError):
        return np.asanyarray(p)


class Hamiltonian(object):
    """
    Full Hamiltonian for an electronics + vibrational system

    Properties
    ----------
    system : hamiltonian.ElectronicHamiltonian
        Object representing the system-part of the Hamiltonian
    bath : bath.Bath
        Object containing the bath information (i.e., correlation function and
        temperature). Each site is assumed to be linearly coupled to an
        identical bath of this form.
    dipoles : np.ndarray, optional
        n x 3 array of dipole moments for each site.
    """
    def __init__(self, system, bath, dipoles=None):
        self.system = system
        self.bath = bath
        self.dipoles = dipoles

    def in_rotating_frame(self, *args, **kwargs):
        """
        Return a new Hamiltonian object with the system shifted into the
        rotating frame
        """
        return type(self)(self.system.in_rotating_frame(*args, **kwargs),
                          self.bath, self.dipoles)

    def sample(self, *args, **kwargs):
        """
        Yield new Hamiltonian objects with a resampled system Hamiltonian
        """
        for sys in self.system.sample(*args, **kwargs):
            yield type(self)(sys, self.bath, self.dipoles)

    @property
    def freq_step(self):
        return self.system.freq_step

    @property
    def time_step(self):
        return self.system.time_step

    def dipole_operator(self, subspace='gef', polarization='x',
                       include_transitions='-+'):
        """
        Return the matrix representation in the given subspace of the requested
        dipole operator
        """
        N = self.system.n_sites
        trans_ops = [transition_operator(n, N, subspace, include_transitions)
                     for n in xrange(N)]
        return np.einsum('nij,nk,k->ij', trans_ops, self.dipoles,
                         polarization_vector(polarization))

    def system_bath_couplings(self, subspace='gef'):
        """
        Return a list of matrix representations in the given subspace of the
        system-bath coupling operators
        """
        N = self.system.n_sites
        return [operator_extend(np.diag(unit_vec(n, N)), subspace)
                for n in xrange(N)]

    def __repr__(self):
        return "{0}(system={1}, bath={2}, dipoles={3})".format(
            type(self).__name__, self.system, self.bath, self.dipoles)


class ElectronicHamiltonian(object):
    """
    Electronic Hamiltonian composed of some finite number of sites

    Properties
    ----------
    H_1 : np.ndarray
        Matrix representation of this hamiltonian in the 1-excitation subspace
    energy_offset : number, optional
        Constant energy offset of the diagonal entries in H_1 from the ground
        state energy.
    disorder_fwhm : number, optional
        Full-width-at-half-maximum of diagonal, Gaussian disorder for
        re-samplings
    ref_system : ElectronicHamiltonian, optional
        A reference to either this object or the reference Hamiltonian from
        which this object was sampled. This attribute is used so that reference
        frequencies are stable.
    sampling_freq_extra : float, optional (default 100)
        Default extra frequency to add to the spread of energies when
        determining the frequency step size automatically.
    """
    def __init__(self, H_1, energy_offset=0, disorder_fwhm=0, ref_system=None,
                 sampling_freq_extra=100.0):
        self.H_1 = H_1
        self.energy_offset = energy_offset
        self.disorder_fwhm = disorder_fwhm
        self.ref_system = ref_system if ref_system is not None else self
        self.sampling_freq_extra = sampling_freq_extra

    @property
    def n_sites(self):
        return len(self.H_1)

    def n_states(self, subspace):
        return len(self.H(subspace))

    @property
    def freq_step(self):
        """
        An appropriate sampling rate, according to the Nyquist theorem, so that
        all frequencies of the Hamiltonian can be resolved
        """
        energies = self.ref_system.E('gef')
        freq_span = energies.max() - energies.min()
        return 2 * (freq_span + self.sampling_freq_extra)

    @property
    def time_step(self):
        return 1.0 / self.freq_step

    @imemoize
    def H(self, subspace):
        """
        Returns this Hamiltonian in matrix in the given subspace
        """
        return operator_extend(self.H_1, subspace)

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

    def ground_state(self, subspace):
        """
        Returns the wave-function for the ground state of this Hamiltonian as
        a state vector in Hilbert space
        """
        state = np.zeros(self.n_states(subspace), dtype=complex)
        if 'g' in subspace:
            state[0] = 1.0
        return state

    @property
    def mean_excitation_freq(self):
        """
        Average excited state transition energy
        """
        return np.mean(self.ref_system.E('e')) + self.energy_offset

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
        H_1 = self.H_1 - shift * np.identity(len(self.H_1))
        ref_system = (self.ref_system.to_rotating_frame(rw_freq)
                      if (self.ref_system is not self) else None)
        return type(self)(H_1, rw_freq, self.disorder_fwhm, ref_system,
                          self.sampling_freq_extra)

    def sample(self, ensemble_size=1, random_seed=None):
        """
        Yields re-samplings of the electronic Hamiltonian with diagonal disorder
        """
        np.random.seed(random_seed)
        for _ in xrange(ensemble_size):
            H_1_prime = (self.H_1 + self.disorder_fwhm * GAUSSIAN_SD_FWHM
                         * np.diag(np.random.randn(self.n_sites)))
            yield type(self)(H_1_prime, self.energy_offset, 0, self.ref_system,
                             self.sampling_freq_extra)

    def __repr__(self):
        return ("{}(H_1={}, energy_offset={} disorder_fwhm={}, ref_system={}, "
                "sampling_freq_extra={})"
                ).format(type(self).__name__, self.H_1, self.energy_offset,
                         self.disorder_fwhm,
                         None if self.ref_system is self else self.ref_system,
                         self.sampling_freq_extra)
