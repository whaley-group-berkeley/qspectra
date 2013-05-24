from abc import ABCMeta, abstractmethod, abstractproperty


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
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def ground_state(self, subspace):
        """
        Ground state for this dynamical model
        """

    @abstractmethod
    def equation_of_motion(self, subspace):
        """
        Return the equation of motion for this dynamical model in the given
        subspace, a function which takes a state vector and returns its first
        time derivative, for use in a numerical integration routine
        """

    @abstractmethod
    def dipole_destroy(self, subspace, polarization):
        """
        Return a dipole annhilation operator that follows the
        SystemFieldOperator API for the given subspace and polarization
        """

    @abstractmethod
    def dipole_create(self, subspace, polarization):
        """
        Return a dipole creation operator that follows the SystemFieldOperator
        API for the given subspace and polarization
        """

    @abstractproperty
    def time_step(self):
        """
        The default time step at which to sample the equation of motion (in the
        rotating frame)
        """


class SystemFieldOperator(object):
    """
    Abstract base class defining the SystemFieldOperator API used by
    spectroscopy simulation methods

    Instances of a SystemFieldOperator class are abstract object whose
    commutator and expectation value can be calculated when applied to arbitrary
    state vectors in a matching subspace used by DynamicalModel objects.

    To implement a new type of system-field operator, inherit from this class
    and override all abstract methods.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def commutator(self, state):
        """
        Returns the commutator of the system-field operator with the given state
        """

    @abstractmethod
    def expectation_value(self, state):
        """
        Returns the expectation value of the system-field operator in the given
        state
        """
