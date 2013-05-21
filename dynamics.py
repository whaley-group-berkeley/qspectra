from abc import ABCMeta, abstractmethod, abstractproperty


class Dynamics(object):
    """
    Abstract base class defining the Dynamics API used by spectroscopy
    simulation methods

    To implement a new type of dynamics, inherit from this class, override
    all abstract methods and define the `rw_freq` attribute in the __init__
    method.

    Attributes
    ----------
    rw_freq : float
        The frequency of the rotating frame in which these dynamics apply.
        Typically, `rw_freq` would be defined by the hamiltonian which provides
        the parameters to initialize a new Dynamics object.
    """
    __meta__ = ABCMeta

    @abstractmethod
    def ground_state(self, subspace):
        """
        Ground state for this type of dynamics
        """

    @abstractmethod
    def equation_of_motion(self, subspace):
        """
        Return the equation of motion these dynamics in the given subspace,
        a function which takes a state and returns its first time derivative,
        for use in a numerical integration routine
        """

    @abstractmethod
    def dipole_destroy(self, polarization, subspace):
        """
        Return a dipole annhilation operator that follows the Operator API for
        the given polarization and subspace
        """

    @abstractmethod
    def dipole_create(self, polarization, subspace):
        """
        Return the dipole creation operator that follows the Operator API for
        the given polarization and subspace
        """

    @abstractproperty
    def time_step(self):
        """
        The default time step at which to sample dynamics (in the rotating
        frame)
        """


class Operator(object):
    """
    Abstract base class defining the Operator API used by spectroscopy
    simulation methods

    Operator classes define an extension of a system operator into an abstract
    object whose commutator and expectation value can be calculated when
    applied to arbitrary the states defined by Dynamics objects.

    To implement a new type of operator, inherit from this class and override
    all abstract methods.
    """
    __meta__ = ABCMeta

    def negative_imag_commutator(self, state):
        """Returns -1j times the commutator of this operator with the state"""
        return -1j * self.commutator(state)

    @abstractmethod
    def commutator(self, state):
        """Returns the commutator of this operator with the state"""

    @abstractmethod
    def expectation_value(self, state):
        """Returns the expectation value of this operator in the state"""
