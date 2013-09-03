from abc import ABCMeta, abstractmethod, abstractproperty

from ..utils import copy_with_new_cache


class DynamicalModel(object):
    """
    Abstract base class defining the DynamicalModel API used by spectroscopy
    simulation methods

    A DynamicalModel instance completely specifies how a system evolves freely
    and in response to applied fields.

    To implement a new type of dynamical model, inherit from this class,
    override all abstract methods and define the `rw_freq` attribute in the
    __init__ method.

    Attributes
    ----------
    rw_freq : float
        The frequency of the rotating frame in which this model applies.
        Typically, `rw_freq` would be defined by the hamiltonian which provides
        the parameters to initialize a new DynamicalModel object.

    Warning
    -------
    In the current implementation of DynamicalModel, it is assumed that you can
    create a modified copy of a dynamical model by merely copying all instance
    variables and replcaing the hamiltonian with a modified hamiltonian. If this
    is not the case for your subclass, you need to override the
    `sample_ensemble` method.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def ground_state(self, liouville_subspace):
        """
        Ground state for this dynamical model
        """

    @abstractmethod
    def equation_of_motion(self, liouville_subspace):
        """
        Return the equation of motion for this dynamical model in the given
        subspace, a function which takes the time and a state vector and returns
        the first time derivative of the state vector, for use in a numerical
        integration routine
        """

    @abstractmethod
    def map_between_subspaces(self, state, from_subspace, to_subspace):
        """
        Given a state vector `state` defined on `from_subspace` in Liouville
        space, return the state mapped onto `to_subspace`.

        If any portion of `to_subspace` is not in `from_subspace`, that portion
        of the state is assumed to be zero.
        """

    @abstractmethod
    def dipole_operator(self, liouv_subspace_map, polarization,
                        include_transitions='-+'):
        """
        Return a dipole operator that follows the SystemOperator API for the
        given liouville_subspace_map, polarization and requested transitions
        """

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

    @abstractproperty
    def time_step(self):
        """
        The default time step at which to sample the equation of motion (in the
        rotating frame)
        """

    def sample_ensemble(self, *args, **kwargs):
        """
        Yields `ensemble_size` re-samplings of this dynamical model by sampling
        the hamiltonian
        """
        for ham in self.hamiltonian.sample_ensemble(*args, **kwargs):
            new_dynamical_model = copy_with_new_cache(self)
            new_dynamical_model.hamiltonian = ham
            yield new_dynamical_model


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
        Returns the left multiplication of the system-field operator with the
        given state
        """

    @abstractmethod
    def expectation_value(self, state):
        """
        Returns the expectation value of the system-field operator in the given
        state
        """
