from abc import ABCMeta, abstractmethod

from ..operator_tools import hilbert_subspace_index
from ..utils import copy_with_new_cache, inspect_repr


class DynamicalModel(object):
    """
    Abstract base class defining the DynamicalModel API used by spectroscopy
    simulation methods

    A DynamicalModel instance completely specifies how a system evolves freely
    and in response to applied fields.

    To implement a new type of dynamical model, inherit from this class and
    override all abstract methods

    Parameters
    ----------
    hamiltonian : hamiltonian.Hamiltonian
        Hamiltonian object specifying the system
    rw_freq : float, optional
        Rotating wave frequency at which to calculate dynamics. By default,
        the rotating wave frequency is chosen from the central frequency
        of the Hamiltonian.
    hilbert_subspace : container, default 'ge'
        Container of any or all of 'g', 'e' and 'f' indicating the maximum
        set of Hilbert subspace on which to calculate the dynamics.
    unit_convert : number, optional
        Unit conversion from energy to time units (default 1).

    Warning
    -------
    In the current implementation of DynamicalModel, it is assumed that you can
    create a modified copy of a dynamical model by merely copying all instance
    variables and replcaing the hamiltonian with a modified hamiltonian. If this
    is not the case for your subclass, you need to override the
    `sample_ensemble` method.
    """
    __metaclass__ = ABCMeta

    def __init__(self, hamiltonian, rw_freq=None, hilbert_subspace='gef',
                 unit_convert=1):
        self.hamiltonian = hamiltonian.in_rotating_frame(rw_freq)
        self.rw_freq = self.hamiltonian.rw_freq
        self.hilbert_subspace = hilbert_subspace
        self.unit_convert = unit_convert

    def __repr__(self):
        return inspect_repr(self)

    @abstractmethod
    def thermal_state(self, liouville_subspace):
        """
        Thermal state for this dynamical model
        """

    @abstractmethod
    def equation_of_motion(self, liouville_subspace, heisenberg_picture=False):
        """
        Return the equation of motion for this dynamical model in the given
        subspace, a function which takes the time and a state vector and returns
        the first time derivative of the state vector, for use in a numerical
        integration routine

        If `heisenberg_picture` is True, returns an equation of motion for
        operators in the Heisenberg picture. If a dynamical model does not
        implement an equation of motion in the Heisenberg, it will raise a
        `NotImplementedError`.
        """

    @abstractmethod
    def map_between_subspaces(self, state, from_subspace, to_subspace):
        """
        Given a state vector `state` defined on `from_subspace` in Liouville
        space, return the state mapped onto `to_subspace`.

        If any portion of `to_subspace` is not in `from_subspace`, that portion
        of the state is assumed to be zero.
        """

    def dipole_operator(self, liouv_subspace_map, polarization,
                        transitions='-+'):
        """
        Return a dipole operator that follows the SystemOperator API for the
        given liouville_subspace_map, polarization and requested transitions
        """
        operator = self.hamiltonian.dipole_operator(self.hilbert_subspace,
                                                    polarization, transitions)
        return self.system_operator(operator, liouv_subspace_map, self)

    def dipole_destroy(self, liouville_subspace_map, polarization):
        """
        Return a dipole annhilation operator that follows the SystemOperator API
        for the given subspace and polarization
        """
        return self.dipole_operator(liouville_subspace_map, polarization, '-')

    def dipole_create(self, liouville_subspace_map, polarization):
        """
        Return a dipole creation operator that follows the SystemOperator
        API for the given liouville_subspace_map and polarization
        """
        return self.dipole_operator(liouville_subspace_map, polarization, '+')

    def sample_ensemble(self, *args, **kwargs):
        """
        Yields `ensemble_size` re-samplings of this dynamical model by sampling
        the hamiltonian
        """
        for ham in self.hamiltonian.sample_ensemble(*args, **kwargs):
            new_dynamical_model = copy_with_new_cache(self)
            new_dynamical_model.hamiltonian = ham
            yield new_dynamical_model

    @property
    def time_step(self):
        """
        The default time step at which to sample the equation of motion (in the
        rotating frame)
        """
        return self.hamiltonian.time_step / self.unit_convert

    def hilbert_subspace_index(self, subspace):
        return self.hamiltonian.hilbert_subspace_index(
            subspace, self.hilbert_subspace)


class SystemOperator(object):
    """
    Abstract base class defining the SystemOperator API used by
    spectroscopy simulation methods

    Instances of a SystemOperator class are abstract object whose
    commutator and expectation value can be calculated when applied to arbitrary
    state vectors in a matching subspace used by DynamicalModel objects.

    To implement a new type of system-field operator, inherit from this class
    and override all abstract methods.
    """
    __metaclass__ = ABCMeta

    def commutator(self, state):
        """
        Returns the commutator of the system-field operator with the given state
        """
        return self.left_multiply(state) - self.right_multiply(state)

    @abstractmethod
    def left_multiply(self, state):
        """
        Returns the left multiplication of the system-field operator with the
        given state
        """

    @abstractmethod
    def right_multiply(self, state):
        """
        Returns the right multiplication of the system-field operator with the
        given state
        """

    @abstractmethod
    def expectation_value(self, state):
        """
        Returns the expectation value of the system-field operator in the given
        state
        """
