# TODO: to improve extendability, move this file into separate subfolder
from abc import ABCMeta, abstractmethod
from numbers import Number

import numpy as np
import scipy.linalg

from .constants import GAUSSIAN_SD_FWHM
from .operator_tools import (transition_operator, operator_extend, unit_vec,
                             tensor, extend_vib_operator, vib_create,
                             vib_annihilate)
from .polarization import polarization_vector, random_rotation_matrix
from .utils import imemoize, memoized_property, check_random_state


class HamiltonianError(Exception):
    """
    Error class for Hamiltonian errors
    """

class Hamiltonian(object):
    """
    Parent class for Hamiltonian objects
    """
    __metaclass__ = ABCMeta

    def __init__(self, energy_offset_source=None):
        # used to keep track of the original, unperturbed Hamiltonian even
        # after calling `sample_ensemble`:
        self.ref_system = self
        # used to keep track of original transition energies even after
        # transforming to a rotating frame:
        if energy_offset_source is None:
            self._energy_offset = 0
            self._energy_offset_source = self 
        else:
            self._energy_offset_source = energy_offset_source

    @property
    def energy_offset(self):
        return self._energy_offset_source._energy_offset

    @energy_offset.setter
    def energy_offset(self, value):
        if self._energy_offset_source is not self:
            raise HamiltonianError('cannot set `energy_offset` for this '
                                   'Hamiltonian directly')
        self._energy_offset = value

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
        Returns the eigensystem solution E, U for the system part of this
        Hamiltonian in the given subspace
        """
        E, U = scipy.linalg.eigh(self.H(subspace))
        return (E, U)

    def E(self, subspace):
        """
        Returns the eigen-energies of the system part of this Hamiltonian in the
        given subspace
        """
        return self.eig(subspace)[0]

    def U(self, subspace):
        """
        Returns the matrix which transform the system part of this Hamiltonian
        from the site to the energy eigen-basis.
        """
        return self.eig(subspace)[1]

    @property
    def mean_excitation_freq(self):
        """
        Average excited state transition energy
        """
        # should remain fixed under sampling and rotating frame transformations
        return np.mean(self.ref_system.E('e')) + self.ref_system.energy_offset

    @property
    def freq_step(self):
        """
        An appropriate sampling rate, according to the Nyquist theorem, so that
        all frequencies of the Hamiltonian can be resolved

        Note: If this frequency is very high, you probably need to transform to
        the rotating frame first.
        """
        # should remain fixed under sampling but NOT under rotating frame
        # transformations
        energies = self.ref_system.E('gef')
        freq_span = energies.max() - energies.min()
        return 2 * (freq_span + self.energy_spread_extra)

    @property
    def time_step(self):
        """
        An appropriate sampling time step, according to the Nyquist theorem, so
        that all frequencies of the Hamiltonian can be resolved

        Note: If this time-step is very short, you probably need to transform to
        the rotating frame first.
        """
        return 1.0 / self.freq_step


def diagonal_gaussian_disorder(fwhm, n_sites):
    def disorder(random_state):
        return np.diag((fwhm * GAUSSIAN_SD_FWHM) * random_state.randn(n_sites))
    return disorder

 
class ElectronicHamiltonian(Hamiltonian):
    """
    Hamiltonian for an electronic system with coupling to an external field
    and an identical bath at each pigment

    Properties
    ----------
    H_1exc : np.ndarray
        Matrix representation of this Hamiltonian in the 1-excitation subspace
    bath : bath.Bath, optional
        Object containing the bath information (i.e., correlation function and
        temperature). Each site is assumed to be linearly coupled to an
        identical bath of this form.
    dipoles : np.ndarray, optional
        n x 3 array of dipole moments for each site.
    disorder : number or function, optional
        Full-width-at-half-maximum of diagonal, Gaussian static disorder
        (independently sampled as each site) or a function which generates new
        examples of static disorder. This argument controls how to generate
        new samples with `sample_ensemble` method. By default (`disorder=None`),
        no static disorder is added.

        If a function (or other callable), it should take a
        `np.random.RandomState` object and return an array which can be added to
        `H_1exc` to provide a new sample of static disorder. For example, to
        produce Gaussian static disorder with standard deviation 100 for system
        with two sites, you could write:
            ```
            def disorder(random_state):
                return np.diag(100 * random_state.randn(2))
            ```
        NOTE: Only use methods of the `random_state` object to generate random
        numbers in your custom function. Otherwise, your random ensemble will
        not be reproducible, which may add noise when you calculate ensemble
        averages with the spectroscopy methods.
    random_seed : int, optional (default 0)
        Random seed used to produce reproducible sampling with the
        `sample_ensemble` method. Must be a non-negative integer or other valid
        input for np.random.RandomState.
    energy_spread_extra : float, optional (default 100)
        Default extra frequency to add to the spread of energies when
        determining the frequency step size automatically. To avoid unnecessary
        work when calculating quantities like correlation functions, this
        constant should be set to roughly the width of inhomogeneous broadening
        or the dephasing rate. Units should match `H_1exc`.

    See also
    --------
    np.random.RandomState
    """
    def __init__(self, H_1exc, bath=None, dipoles=None, disorder=None,
                 random_seed=0, energy_spread_extra=100.0):
        self.H_1exc = np.asarray(H_1exc)
        self.bath = bath
        self.dipoles = np.asarray(dipoles) if dipoles is not None else None
        self.disorder = disorder
        self.random_seed = random_seed
        self.energy_spread_extra = energy_spread_extra
        # used by various dynamics methods to determine indices:
        self.n_vibrational_states = 1
        super(ElectronicHamiltonian, self).__init__()

    @property
    def n_sites(self):
        return len(self.H_1exc)

    @imemoize
    def H(self, subspace):
        """
        Returns the system Hamiltonian in the given Hilbert subspace as a matrix
        """
        return operator_extend(self.H_1exc, subspace)

    @imemoize
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

        By default, sets the rotating frame frequency to the mean excitation
        frequency. This method is idempotent: applying it twice will return the
        same Hamiltonian as applying it once.

        Note: The `in_rotating_frame` and `sample_ensemble` methods are designed
        carefully so that they commute -- it should not matter in which order
        they are applied.
        """
        if rw_freq is None:
            rw_freq = self.mean_excitation_freq
        H_1exc = self.H_1exc - ((rw_freq - self.energy_offset)
                                * np.identity(len(self.H_1exc)))
        ham = type(self)(H_1exc, self.bath, self.dipoles, self.disorder,
                         self.random_seed, self.energy_spread_extra)
        ham.energy_offset = rw_freq
        if self.ref_system is not self:
            ham.ref_system = self.ref_system.in_rotating_frame(rw_freq)
        return ham

    def sample_ensemble(self, ensemble_size=1, random_orientations=False):
        """
        Yields `ensemble_size` re-samplings of this Hamiltonian over disorder

        Each re-sampled Hamiltonian is another valid Hamiltonian, except
        its default rotating wave frequency, time step and frequency step all
        match the parent Hamiltonian. This guarantees that all the time and
        frequency ticks match when simulating an ensemble generated from this
        method.

        Hamiltonians produced by this method have disorder set to `None`, so
        if you use their `sample_ensemble` method, you will not produce any
        additional static disorder, although you could randomize their
        orientations.

        Note: The `in_rotating_frame` and `sample_ensemble` methods are designed
        carefully so that they commute -- it should not matter in which order
        they are applied.
        """
        if self.disorder is None:
            disorder_func = lambda x: 0
        elif isinstance(self.disorder, Number):
            disorder_func = diagonal_gaussian_disorder(self.disorder,
                                                       self.n_sites)
        else:
            disorder_func = self.disorder
        seed = list(np.atleast_1d(self.random_seed))
        for n in xrange(ensemble_size):
            random_state = check_random_state(seed + [n])
            H_1exc = self.H_1exc + disorder_func(random_state)
            if random_orientations:
                dipoles = np.einsum('mn,in->im',
                                    random_rotation_matrix(random_state),
                                    self.dipoles)
            else:
                dipoles = self.dipoles
            # note: sampled Hamiltonians should have no static disorder
            ham = type(self)(H_1exc, self.bath, dipoles, None, None,
                             self.energy_spread_extra)
            ham.ref_system = self
            yield ham

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

    def number_operator(self, site, subspace='gef'):
        """
        Returns the number operator a_n^\dagger a_n for site n
        """
        return operator_extend(
            np.diag(unit_vec(site, self.n_sites, dtype=float)), subspace)

    def system_bath_couplings(self, subspace='gef'):
        """
        Return a list of matrix representations in the given subspace of the
        system-bath coupling operators
        """
        if self.bath is None:
            raise HamiltonianError('bath undefined')
        return np.array([self.number_operator(n, subspace)
                         for n in xrange(self.n_sites)])


def thermal_state(hamiltonian_matrix, temperature):
    """
    Given a Hamiltonian in matrix form and a temperature, return the thermal
    density matrix

    Parameters
    ----------
    hamiltonian_matrix : np.ndarray
        Hamiltonian as an explicit matrix
    temperature : float
        Bath temperature, in the same units as the Hamiltonian

    Returns
    -------
    rho : np.ndarray
        Density matrix for thermal equilibrium
    """
    rho = scipy.linalg.expm(-hamiltonian_matrix / temperature)
    return rho / np.trace(rho)


class VibronicHamiltonian(Hamiltonian):
    """
    Hamiltonian which extends an electronic Hamiltonian to include explicit
    vibrations

    Properties
    ----------
    electronic : ElectronicHamiltonian
        Object which represents the electronic part of the Hamiltonian,
        including its bath.
    n_vibrational_levels : np.ndarray
        Array giving the number of energy levels to include with each
        vibration.
    vib_energies : np.ndarray
        Array giving the energies of the vibrational modes.
    elec_vib_couplings : np.ndarray
        2D array giving the electronic-vibrational couplings [c_{nm}], where
        the coupling operators are in the form:
        c_{nm}*|n><n|*(b(m) + b(m)^\dagger),
        where |n> is the singly excited electronic state of site n in the full
        singly excited subspace, and b(m) and b(m)^\dagger are the
        vibrational annihilation and creation operators for vibration m.
    """
    def __init__(self, electronic, n_vibrational_levels, vib_energies,
                 elec_vib_couplings):
        self.electronic = electronic
        self.energy_spread_extra = self.electronic.energy_spread_extra
        self.bath = self.electronic.bath
        self.n_sites = self.electronic.n_sites
        self.n_vibrational_levels = np.asarray(n_vibrational_levels)
        self.vib_energies = np.asarray(vib_energies)
        self.elec_vib_couplings = np.asarray(elec_vib_couplings)
        super(VibronicHamiltonian, self).__init__(self.electronic)

    @memoized_property
    def n_vibrational_states(self):
        """
        Returns the total number of vibrational states in the full vibrational
        subspace (i.e. the dimension of the full vibrational subspace)
        """
        return np.prod(self.n_vibrational_levels)

    @memoized_property
    def H_vibrational(self):
        """
        Returns the Hamiltonian of the vibrations included explicitly in this
        model
        """
        H_vib = np.diag(np.zeros(self.n_vibrational_states))
        for m, (num_levels, vib_energy) in \
                enumerate(zip(self.n_vibrational_levels, self.vib_energies)):
            vib_operator = np.diag(np.arange(num_levels))
            H_vib += (vib_energy
                      * extend_vib_operator(self.n_vibrational_levels, m,
                                            vib_operator))
        return H_vib

    def H_electronic_vibrational(self, subspace='gef'):
        """
        Returns the electronic-vibrational coupled part of the Hamiltonian,
        given by
        H_{el-vib} = sum_{n,m} c_{nm}*|n><n|*(b(m) + b(m)^\dagger)
        where |n> is the singly excited electronic state of site n in the full
        singly excited subspace, and b(m) and b(m)^\dagger are the
        annihilation and creation operators for vibrational mode m
        """
        H_el_vib = np.diag(np.zeros(self.electronic.n_states(subspace)
                                    * self.n_vibrational_states))
        for i in np.arange(self.electronic.n_sites):
            el_operator = self.electronic.number_operator(i, subspace)
            for m, num_levels in enumerate(self.n_vibrational_levels):
                vib_operator = (vib_annihilate(num_levels)
                                + vib_create(num_levels))
                H_el_vib += (self.elec_vib_couplings[i, m]
                             * tensor(el_operator,
                                      extend_vib_operator(
                                          self.n_vibrational_levels,
                                          m, vib_operator)))
        return H_el_vib

    @imemoize
    def H(self, subspace='gef'):
        """
        Returns the matrix representation of the system Hamiltonian in the
        given electronic subspace
        """
        return (self.el_to_sys_operator(self.electronic.H(subspace))
                + self.vib_to_sys_operator(self.H_vibrational, subspace)
                + self.H_electronic_vibrational(subspace))

    @imemoize
    def ground_state(self, subspace='gef'):
        if self.bath is None:
            raise HamiltonianError('bath needs to be defined determine the '
                                   'equilibrium density matrix in the '
                                   'electronic ground state')
        return np.kron(self.electronic.ground_state(subspace),
                       thermal_state(self.H_vibrational,
                                     self.bath.temperature))

    @imemoize
    def in_rotating_frame(self, rw_freq=None):
        """
        Returns a new Hamiltonian shifted to the rotating frame at the given
        frequency

        By default, sets the rotating frame frequency to the mean excitation
        frequency of the electronic part of the Hamiltonian.
        """
        return type(self)(self.electronic.in_rotating_frame(rw_freq),
                          self.n_vibrational_levels, self.vib_energies,
                          self.elec_vib_couplings)

    def sample_ensemble(self, *args, **kwargs):
        """
        Yields `ensemble_size` re-samplings of this Hamiltonian over disorder

        Passes on all arguments to the contained electronic Hamiltonian; each
        re-sampling replaces the electronic Hamiltonian.
        """
        for elec in self.electronic.sample_ensemble(*args, **kwargs):
            ham = type(self)(elec, self.n_vibrational_levels, self.vib_energies,
                             self.elec_vib_couplings)
            ham.ref_system = self
            yield ham

    def el_to_sys_operator(self, el_operator):
        """
        Extends the electronic operator el_operator, which may be in an
        electronic subspace, into a system operator in that subspace
        """
        return tensor(el_operator, np.eye(self.n_vibrational_states))

    def vib_to_sys_operator(self, vib_operator, subspace='gef'):
        """
        Extends the vibrational operator vib_operator, which may be in a
        vibrational subspace, into a system operator in that subspace
        and in the given electronic subspace
        """
        return tensor(np.eye(self.electronic.n_states(subspace)), vib_operator)

    def dipole_operator(self, *args, **kwargs):
        """
        Return the matrix representation in the given subspace of the requested
        dipole operator
        """
        return self.el_to_sys_operator(
            self.electronic.dipole_operator(*args, **kwargs))

    def system_bath_couplings(self, *args, **kwargs):
        """
        Return a list of matrix representations in the given subspace of the
        system-bath coupling operators
        """
        return self.el_to_sys_operator(
            self.electronic.system_bath_couplings(*args, **kwargs))
