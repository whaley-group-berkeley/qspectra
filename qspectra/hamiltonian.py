# TODO: to improve extendability, move this file into separate subfolder
from abc import ABCMeta, abstractmethod
from numbers import Number
import warnings

import numpy as np
import scipy.linalg

from .constants import GAUSSIAN_SD_FWHM
from .operator_tools import (transition_operator, operator_extend, unit_vec,
                             tensor, extend_vib_operator, vib_create,
                             vib_annihilate, hilbert_subspace_index)
from .polarization import polarization_vector, random_rotation_matrix
from .utils import imemoize, memoized_property, check_random_state, inspect_repr


def check_hermitian(matrix):
    matrix = np.asarray(matrix)
    if not np.allclose(matrix.conj().T, matrix):
        raise ValueError('matrix input must to be Hermitian')
    return matrix


def ground_state(hamiltonian_matrix):
    """
    Given a Hamiltonian in matrix form, return its ground state

    Parameters
    ----------
    hamiltonian_matrix : np.ndarray
        Hamiltonian as an explicit matrix

    Returns
    -------
    rho : np.ndarray
        Density matrix for the ground state
    """
    E, U = scipy.linalg.eigh(hamiltonian_matrix)
    # note: U will be real valued, since H is hermitian, so there is no need to
    # use np.conj()
    # also note: the energies E from eigh are sorted in ascending order, so we
    # know that E[0] is the minimum
    rho = np.mean([np.outer(U[:, i], U[:, i]) for i in xrange(len(U))
                   if E[i] == E[0]], axis=0)
    return rho.astype(complex)


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
    if temperature > 0:
        hamiltonian_matrix = np.asarray(hamiltonian_matrix)
        rho = scipy.linalg.expm(-hamiltonian_matrix / float(temperature))
        trace = np.trace(rho)
        if trace == 0 or np.isnan(rho).any():
            raise OverflowError(('temperature=%s too low to reliably calculate '
                                 'thermal_state; raise it or set it to zero '
                                 '(in which case ground_state is substituted')
                                 % temperature)
        rho /= trace
    else:
        rho = ground_state(hamiltonian_matrix)
    return rho.astype(complex)

def add_braket(basis_labels):
    braket_labels = []
    for label in basis_labels:
        if isinstance(label, basestring) or not np.iterable(label):
            braket_label = '|{}>'.format(label)
        else:
            braket_label = ''.join('|{}>'.format(i) for i in label)
        braket_labels.append(braket_label)
    return braket_labels

class HamiltonianError(Exception):
    """
    Error class for Hamiltonian errors
    """


class Hamiltonian(object):
    """
    Parent class for Hamiltonian objects

    At a minimum, subclasses should implement a new `H` method which returns the
    Hamiltonian as an explicit matrix.
    """
    __metaclass__ = ABCMeta

    def __init__(self, energy_spread_extra=None, site_labels=None):
        self.energy_spread_extra = energy_spread_extra
        # keep track of the non-sampled and non-rotating version of this
        # Hamiltonian for the `in_rotating_frame` and `sample` methods:
        self._not_sampled = self
        self._not_rotating = self
        # keep track of the rotating wave frequency, so it's possible to check
        # what rotating wave frquency has been set:
        self.rw_freq = 0
        self.site_labels = site_labels

    @property
    def _original(self):
        """
        Reference to the non-sampled and non-rotating version of this
        Hamiltonian
        """
        return self._not_rotating._not_sampled

    def __repr__(self):
        return inspect_repr(self)

    def __eq__(self, other):
        return self._eq(other, max_depth=1)

    # set this to True if a subclass implements a meaningful `_eq` method
    _implements_eq = False

    def _eq(self, other, max_depth):
        if not self._implements_eq:
            # fall back on comparison by memory address; otherwise, by default
            # _eq thinks all subclass Hamiltonians are equal
            return self is other
        else:
            # recursively check reference Hamiltonians for for equality until
            # max_depth is 0, at which point assume equality
            # TODO: prove that max_depth=1 is sufficient in all cases
            return (not max_depth or
                    (self.rw_freq == other.rw_freq and
                     self._not_sampled._eq(other._not_sampled, max_depth - 1) and
                     self._not_rotating._eq(other._not_rotating, max_depth - 1)))

    def __ne__(self, other):
        return not self == other

    @abstractmethod
    def H(self, subspace):
        """
        Returns the system Hamiltonian in the given Hilbert subspace as a matrix
        """

    @imemoize
    def ground_state(self, subspace):
        """
        Returns the ground state of this Hamiltonian as a density operator
        """
        return ground_state(self._not_rotating.H(subspace))

    @imemoize
    def thermal_state(self, subspace):
        """
        Returns the thermal state of this Hamiltonian as a density operator

        If there is no bath or the bath does not define a temperature, the
        temperature is assumed to be zero.
        """
        try:
            temperature = self._not_rotating.bath.temperature
        except AttributeError:
            temperature = 0
        return thermal_state(self._not_rotating.H(subspace), temperature)

    @imemoize
    def in_rotating_frame(self, rw_freq=None):
        """
        Returns a new Hamiltonian shifted to the rotating frame at the given
        frequency

        This method is idempotent: applying it twice will return the
        same Hamiltonian as applying it once.

        Note: The `in_rotating_frame` and `sample` methods are designed
        carefully so that they commute -- it should not matter in which order
        they are applied.

        Parameters
        ----------
        rw_freq : float, optional
            Frequency of the rotating frame to which to transformation this
            Hamiltonian. By default, sets the rotating frame frequency to the
            transition energy.

        Returns
        -------
        ham : Hamiltonian
            New instance of this Hamiltonian type shifted to the rotating frame.
        """
        if rw_freq is None:
            rw_freq = self._original.transition_energy
        ham = self._not_rotating._in_rotating_frame(rw_freq)
        ham._not_rotating = self._not_rotating
        if self._not_sampled is not self:
            ham._not_sampled = self._not_sampled.in_rotating_frame(rw_freq)
        ham.rw_freq = rw_freq
        return ham

    def _in_rotating_frame(self, rw_freq):
        """
        Override this method to implement rotating frame transformations for
        this Hamiltonian.

        Don't call this method directly: use `in_rotating_frame` instead, which
        takes care of some important book-keeping.
        """
        raise NotImplementedError(('%s does not implement rotating frame '
                                   'transformations') % type(self).__name__)

    def sample_ensemble(self, ensemble_size=1, random_orientations=False):
        """
        Yields `ensemble_size` samplings of this Hamiltonian over static
        disorder

        Note: The ensemble returned by this method is not stochastic. The first
        n ensemble members will always be the same.
        """
        for n in xrange(ensemble_size):
            yield self.sample(n, random_orientations)

    def sample(self, n=None, random_orientations=False):
        """
        Produce the nth sampled Hamiltonian over static disorder

        Each re-sampled Hamiltonian is another valid Hamiltonian, except
        its default rotating wave frequency, time step and frequency step all
        match the parent Hamiltonian. This guarantees that all the time and
        frequency ticks match between different ensemble members sampled from
        the same Hamiltonian.

        If you resample a Hamiltonian produced by this method, it will add the
        static disorder to the original Hamiltonian. Thus this method is
        idempotent in a particular sense: the distribution of Hamiltonians
        sampled from sampled Hamiltonians is equivalent to the distribution
        of Hamiltonians sampled directly (although of course the particular
        ensemble members will be different).

        Note: The `in_rotating_frame` and `sample` methods are designed
        carefully so that they commute -- it should not matter in which order
        they are applied.

        Parameters
        ----------
        n : int, optional
            Number to identify the desired ensemble member. By default, this
            number is chosen at random.
        random_orientations : bool, optional
            If True, randomized the orientation of of the sampled Hamiltonian.

        Returns
        -------
        ham : Hamiltonian
            New instance of this Hamiltonian type sampled over static disorder
        """
        if n is None:
            n = np.random.randint(2 ** 30)
        ham = self._not_sampled._sample(n, random_orientations)
        if self._not_rotating is not self:
            ham._not_rotating = self._not_rotating.sample(n, random_orientations)
        ham._not_sampled = self._not_sampled
        ham.rw_freq = self.rw_freq
        return ham

    def _sample(self, n, random_orientations):
        """
        Override this method to implement sampling over static disorder for this
        Hamiltonian

        Don't call this method directly: use `sample` instead, which takes care
        of some important book-keeping.

        Note: This method should be not stochastic! Use a random seed so that
        the nth sampled Hamiltonian is always the same.
        """
        raise NotImplementedError('%s does not implement ensemble sampling'
                                  % type(self).__name__)

    def dipole_operator(self, subspace='gef', polarization='x',
                        transitions='-+'):
        """
        Return the matrix representation in the given subspace of the requested
        dipole operator
        """
        raise NotImplementedError('%s does not implement dipole operators'
                                  % type(self).__name__)

    def system_bath_couplings(self, subspace='gef'):
        """
        Return a list of matrix representations in the given subspace of the
        system-bath coupling operators
        """
        raise NotImplementedError('%s does not implement system-bath couplings'
                                  % type(self).__name__)

    def n_states(self, subspace):
        return len(self.H(subspace))

    @imemoize
    def eig(self, subspace):
        """
        Returns the eigensystem solution E, U for the system part of this
        Hamiltonian in the given subspace

        The eigenvalues E arrive in ascending order by subspace and then by
        value, i.e., 0-excitation states followed by 1-excitation states
        followed by 2-excitation states.
        """
        # solve the eigenvalue problem in the non-rotating basis to preserve
        # the ordering between blocks for each number of excitations (eigh
        # guarantees eigenvalues are returned in ascending order)
        E, U = scipy.linalg.eigh(self._not_rotating.H(subspace))
        if 'e' in subspace:
            E[self.hilbert_subspace_index('e', subspace)] -= self.rw_freq
        if 'f' in subspace:
            E[self.hilbert_subspace_index('f', subspace)] -= 2 * self.rw_freq
        return (E, U)

    def E(self, subspace):
        """
        Returns the eigen-energies of the system part of this Hamiltonian in the
        given subspace, in ascending order
        """
        return self.eig(subspace)[0]

    def U(self, subspace):
        """
        Returns the matrix which transform the system part of this Hamiltonian
        from the site to the energy eigen-basis
        """
        return self.eig(subspace)[1]

    @property
    def transition_energy(self):
        """
        A single number estimate of the excited state transition energy
        """
        return np.mean(self.E('e'))

    @property
    def freq_step(self):
        """
        An appropriate sampling rate, according to the Nyquist theorem, so that
        all frequencies of the Hamiltonian can be resolved

        This property remains fixed under ensemble sampling.

        Note: If this frequency is very high, you probably need to transform to
        the rotating frame first.
        """
        energies = self._not_sampled.E('gef')
        if self.energy_spread_extra is None:
            freq_extra = 0.01 * self._original.transition_energy
        else:
            freq_extra = self.energy_spread_extra
        freq_max = max(energies.max(), -energies.min()) + freq_extra
        return 2 * freq_max

    @property
    def time_step(self):
        """
        An appropriate sampling time step, according to the Nyquist theorem, so
        that all frequencies of the Hamiltonian can be resolved

        This property remains fixed under ensemble sampling.

        Note: If this time-step is very short, you probably need to transform to
        the rotating frame first.
        """
        return 1.0 / self.freq_step

    def basis_labels(self, subspace, braket=False):
        return add_braket(self.site_labels) if braket else self.site_labels

    def H_dataframe(self, subspace, braket=False):
        """
        Returns the Hamiltonian matrix wrapped in a Pandas DataFrame. Useful for
        pretty printing if the basis labels are defined.
        """
        import pandas as pd
        labels = self.basis_labels(subspace, braket)
        matrix = self.H(subspace)
        return pd.DataFrame(matrix, columns=labels, index=labels)

    def U_dataframe(self, subspace, braket=False):
        """
        Returns the eigenvectors wrapped in a Pandas DataFrame. Useful for
        pretty printing if the basis labels are defined.
        """
        import pandas as pd
        labels = self.basis_labels(subspace, braket)
        matrix = self.U(subspace)
        energies = self.E(subspace)
        return pd.DataFrame(matrix, columns=energies, index=labels)

    def hilbert_subspace_index(self, subspace, all_subspaces):
        """
        Given a Hilbert subspace 'g', 'e' or 'f' and the set of all subspaces on
        which a state is defined, returns a slice object to select all elements in
        the given subspace

        Examples
        --------
        >>> ham = ElectronicHamiltonian(np.eye(2))
        >>> ham.hilbert_subspace_index('g', 'gef')
        slice(0, 1)
        >>> ham.hilbert_subspace_index('e', 'gef', 2)
        slice(1, 3)
        >>> ham.hilbert_subspace_index('f', 'gef', 2)
        slice(3, 4)
        """
        return hilbert_subspace_index(subspace, all_subspaces, self.n_sites,
                                      self.n_vibrational_states)


def diagonal_gaussian_disorder(fwhm, n_sites):
    def disorder(random_state):
        return np.diag((fwhm * GAUSSIAN_SD_FWHM) * random_state.randn(n_sites))
    return disorder


class ElectronicHamiltonian(Hamiltonian):
    """
    Hamiltonian for an electronic system with coupling to an external field
    and an identical bath at each pigment

    Parameters
    ----------
    H_1exc : np.ndarray
        Matrix representation of this Hamiltonian in the 1-excitation
        subspace
    bath : bath.Bath, optional
        Object containing the bath information (i.e., correlation function
        and temperature). Each site is assumed to be linearly coupled to an
        identical bath of this form.
    dipoles : np.ndarray, optional
        n x 3 array of dipole moments for each site.
    disorder : number or function, optional
        Full-width-at-half-maximum of diagonal, Gaussian static disorder
        (independently sampled as each site) or a function which generates
        new examples of static disorder. This argument controls how to
        generate new samples with `sample_ensemble` method. By default
        (`disorder=None`), no static disorder is added.

        If a function (or other callable), it should take a
        `np.random.RandomState` object and return an array which can be
        added to `H_1exc` to provide a new sample of static disorder. For
        example, to produce Gaussian static disorder with standard deviation
        100 for system with two sites, you could write::

            def disorder(random_state):
                return np.diag(100 * random_state.randn(2))

        NOTE: Only use methods of the `random_state` object to generate
        random numbers in your custom function. Otherwise, your random
        ensemble will not be reproducible, which may add noise when you
        calculate ensemble averages with the spectroscopy methods.
    random_seed : int, optional
        Random seed used to produce reproducible sampling with the
        `sample_ensemble` method. Must be a non-negative integer or other
        valid input for np.random.RandomState.
    energy_spread_extra : float, optional
        Default extra frequency to add to the spread of energies when
        determining the frequency step size automatically. To avoid
        unnecessary work when calculating quantities like correlation
        functions, this constant should be set to roughly the width of
        inhomogeneous broadening or the dephasing rate. Units should match
        `H_1exc`. By default, this is set to one percent of the average
        excited state transition energy.
    """
    def __init__(self, H_1exc, bath=None, dipoles=None, disorder=None,
                 random_seed=0, energy_spread_extra=None, site_labels=None):
        self.H_1exc = check_hermitian(H_1exc)
        self.bath = bath
        self.dipoles = np.asarray(dipoles) if dipoles is not None else None
        self.disorder = disorder
        self.random_seed = random_seed
        # used by various dynamics methods to determine indices:
        self.n_vibrational_states = 1
        super(ElectronicHamiltonian, self).__init__(energy_spread_extra, site_labels)

    _implements_eq = True

    def _eq(self, other, max_depth):
        return (np.all(self.H_1exc == other.H_1exc) and
                self.bath == other.bath and
                np.all(self.dipoles == other.dipoles) and
                self.disorder == other.disorder and
                self.random_seed == other.random_seed and
                self.energy_spread_extra == other.energy_spread_extra and
                super(ElectronicHamiltonian, self)._eq(other, max_depth))

    @property
    def n_sites(self):
        return len(self.H_1exc)

    @imemoize
    def H(self, subspace):
        """
        Returns the system Hamiltonian in the given Hilbert subspace as a matrix
        """
        return operator_extend(self.H_1exc, subspace)

    def _in_rotating_frame(self, rw_freq):
        H_1exc = self.H_1exc - rw_freq * np.identity(self.n_sites)
        return type(self)(H_1exc, self.bath, self.dipoles, self.disorder,
                          self.random_seed, self.energy_spread_extra)

    def _sample(self, n, random_orientations):
        if self.disorder is None:
            if not random_orientations:
                warnings.warn('called sample with `disorder=None` and '
                              '`random_orientations=False`: sampled '
                              'Hamiltonian is identical to original',
                              RuntimeWarning, stacklevel=2)
            disorder_func = lambda x: 0
        elif isinstance(self.disorder, Number):
            disorder_func = diagonal_gaussian_disorder(self.disorder,
                                                       self.n_sites)
        else:
            disorder_func = self.disorder

        random_seed = list(np.atleast_1d(self.random_seed)) + [n]
        random_state = check_random_state(random_seed)

        H_1exc = self.H_1exc + disorder_func(random_state)
        if random_orientations:
            dipoles = np.einsum('mn,in->im',
                                random_rotation_matrix(random_state),
                                self.dipoles)
        else:
            dipoles = self.dipoles
        return type(self)(H_1exc, self.bath, dipoles, self.disorder,
                          random_seed, self.energy_spread_extra)

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

    def basis_labels(self, subspace='gef', braket=False):
        """
        If custom labels are used but the ground state is included, then the
        label "g" is used to represent the ground state. If site_labels is None,
        then the Fock states are used (000, 100, 010, 001 ...)
        """
        labels = self._get_Fock_basis_labels(subspace, self.site_labels)
        return add_braket(labels) if braket else labels

    def _get_Fock_basis_labels(self, subspace, labels=None):
        labels_1exc = [10 ** (self.n_sites - i - 1) for i in range(self.n_sites)]
        labels_full = np.diag(operator_extend(np.diag(labels_1exc), subspace))
        label_indices = [str(i).zfill(self.n_sites) for i in labels_full]

        if labels is None:
            return label_indices
        else:
            custom_labels = [','.join([label for i, label in enumerate(labels)
                             if state[i] == '1']) for state in label_indices]
            custom_labels[0] = 'g'
            return custom_labels

class VibronicHamiltonian(Hamiltonian):
    """
    Hamiltonian which extends an electronic Hamiltonian to include explicit
    vibrations

    Parameters
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
        where |n> is the singly excited electronic state of site n in the
        full singly excited subspace, and b(m) and b(m)^\dagger are the
        vibrational annihilation and creation operators for vibration m.
    energy_spread_extra : float, optional
        Default extra frequency to add to the spread of energies when
        determining the frequency step size automatically. To avoid
        unnecessary work when calculating quantities like correlation
        functions, this constant should be set to roughly the width of
        inhomogeneous broadening or the dephasing rate. Units should match
        `H_1exc`. By default, this is set to one percent of the average
        excited state transition energy.
    """
    def __init__(self, electronic, n_vibrational_levels, vib_energies,
                 elec_vib_couplings, energy_spread_extra=None, site_labels=None):
        self.electronic = electronic
        self.energy_spread_extra = self.electronic.energy_spread_extra
        self.bath = self.electronic.bath
        self.n_sites = self.electronic.n_sites
        self.n_vibrational_levels = np.asarray(n_vibrational_levels)
        self.vib_energies = np.asarray(vib_energies)
        self.elec_vib_couplings = np.asarray(elec_vib_couplings)
        super(VibronicHamiltonian, self).__init__(energy_spread_extra, site_labels)

    _implements_eq = True

    def _eq(self, other, max_depth):
        return (self.electronic == other.electronic and
                np.all(self.n_vibrational_levels ==
                       other.n_vibrational_levels) and
                np.all(self.vib_energies == other.vib_energies) and
                np.all(self.elec_vib_couplings == other.elec_vib_couplings) and
                super(VibronicHamiltonian, self)._eq(other, max_depth))

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

    def _in_rotating_frame(self, rw_freq):
        return type(self)(self.electronic.in_rotating_frame(rw_freq),
                          self.n_vibrational_levels, self.vib_energies,
                          self.elec_vib_couplings, self.energy_spread_extra)

    def _sample(self, n, random_orientations):
        return type(self)(self.electronic.sample(n, random_orientations),
                          self.n_vibrational_levels, self.vib_energies,
                          self.elec_vib_couplings, self.energy_spread_extra)

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

    def vib_basis_labels(self):
        vib_label_operator = np.diag(np.zeros(self.n_vibrational_states))
        num_sites = len(self.n_vibrational_levels)
        for m, num_levels in enumerate(self.n_vibrational_levels):
            index = 10 ** (num_sites - m - 1)
            vib_operator = np.diag(np.arange(num_levels))
            temp =  extend_vib_operator(self.n_vibrational_levels, m, vib_operator)
            vib_label_operator += index * temp
        vib_labels = np.diag(vib_label_operator)
        return [('{:0' + str(num_sites) + '}').format(int(i)) for i in vib_labels]

    def basis_labels(self, subspace='gef', braket=False):
        """
        If double excitations are requested, then default to using the Fock basis
        If custom labels are used but the ground state is included, then the
        label "gs" is prepended.

        Vibronic basis labels are returned as a list of tuples:
        [(elec_basis_label, vib_basis_label), ]
        """
        elec_labels = self.electronic.basis_labels(subspace)
        vib_labels = self.vib_basis_labels()
        labels = [(e, v) for e in elec_labels for v in vib_labels]
        return add_braket(labels) if braket else labels
