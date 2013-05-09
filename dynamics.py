from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np

from operator_tools import (S_commutator, density_subset,
                            diag_vec, unit_vec, den_to_vec)
from utils import memoize, memoized_property, odeint


class RestrictedDynamics(object):
    def __init__(self, dynamics, parts):
        self.dynamics = dynamics
        self.parts = parts
        self.vec_selected = density_subset(parts,
                                           dynamics.hamiltonian.system.n_sites)
        self.super_op_selected = np.ix_(self.vec_selected, self.vec_selected)

    def restrict_operator(self, op):
        return type(op)(op.operator, self.super_op_selected)

    @memoize
    def dipole_destroy(self, polarization):
        return self.restrict_operator(self.dynamics.dipole_create(polarization))

    @memoize
    def dipole_create(self, polarization):
        return self.restrict_operator(self.dynamics.dipole_create(polarization))


class Dynamics(object):
    __meta__ = ABCMeta

    _restricted_dynamics = RestrictedDynamics
    _ode_defaults = {'rtol': 1e-4,
                     'max_step': 3,
                     'method': 'adams',
                     'return_meta': True}

    def __init__(self, hamiltonian, rw_freq=None, subspace=None,
                 ode_settings=None):
        self.hamiltonian = hamiltonian.to_rotating_frame(rw_freq)
        self.rw_freq = rw_freq
        self.subspace = subspace
        self.ode_settings = (ode_settings if ode_settings is not None
                             else self._ode_defaults)

    def restrict_dynamics(self, state_parts):
        return self._restricted_dynamics(self, state_parts)

    def dipole_operator(self, polarization):
        """Returns the dipole operator from dynamics.hamiltonian"""
        return self.hamiltonian.dipole_operator(polarization, self.subspace)

    @property
    def time_step(self):
        """The time step at which sample dynamics"""
        return self.hamiltonian.time_step

    @abstractmethod
    def from_density_vec(self, vec):
        """Given a vectorized density operator, return the corresponding state
        used internally by this type of dynamics"""

    @abstractmethod
    def to_density_vec(self, vec):
        """Given the state used internally by this type of dynamics, return
        the corresponding vectorized density operator"""

    @abstractproperty
    def ground_state(self):
        """Ground state for this type of dynamics"""

    @memoize
    def dipole_destroy(self, polarization):
        """Returns the dipole annhilation [super-]operator"""
        return self.to_operator(np.triu(self.dipole_operator(polarization)))

    @memoize
    def dipole_create(self, polarization):
        """Returns the dipole creation [super-]operator"""
        return self.to_operator(np.tril(self.dipole_operator(polarization)))

    @abstractmethod
    def to_operator(self, operator):
        """Transforms an system-operator to an operator in space of this type
        of dynamics"""

    @abstractmethod
    def step(self, rho):
        """Returns the time derivative of the state rho"""

    def integrate(self, f=None, initial_state=None, t=None, **ode_settings):
        if f is None:
            f = self.step
        if initial_state is None:
            initial_state = self.ground_state
        if t is None:
            t = np.arange(0, 1001) * self.time_step
        ode_settings = dict(self.ode_settings, **ode_settings)
        return odeint(f, initial_state, t,
                      save=self.to_density_vec,
                      load=self.from_density_vec,
                      **self.ode_settings)


class LiouvilleSpaceDynamics(Dynamics):
    def from_density_vec(self, vec):
        return vec

    def to_density_vec(self, vec):
        return vec

    def to_operator(self, op):
        return LiouvilleSpaceOperator(op)

    @property
    def ground_state(self):
        N = self.hamiltonian.system.n_states(self.subspace)
        return unit_vec(0, N ** 2)

    @property
    def trace(self):
        return diag_vec(self.hamiltonian.system.n_states(self.subspace))


class LiouvilleSpaceOperator(object):
    """Class for operators that act in Liouville space"""
    def __init__(self, operator, selected_indices=None):
        self.operator = operator
        self.selected_indices = (selected_indices
                                 if selected_indices is not None
                                 else slice(None))
    @memoized_property
    def super_commutator(self):
        """Matrix representation of the super-operator for the commutator of
        this operator and any density operator"""
        return S_commutator(self.operator)[self.selected_indices]

    @memoized_property
    def super_neg_im_commutator(self):
        return -1j * S_commutator(self.operator)[self.selected_indices]

    def step(self, vec):
        return self.super_neg_im_commutator.dot(vec)

    def measure(self, vec):
        return den_to_vec(self.operator.T).dot(vec)
