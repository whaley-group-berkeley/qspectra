"""Simulate density matrix dynamics"""
from functools import wraps
from numpy import conj
import numpy as np

from constants import CM_FS, CM_K
from operator_tools import unit_vec, den_to_vec
from pulse import GaussianPulse
from redfield import redfield_evolve
from utils import MetaArray, odeint
try:
    import hierarchy
except ImportError:
    pass


_ODE_DEFAULTS = {'rtol': 1e-4,
                 'max_step': 3,
                 'method': 'adams',
                 'return_meta': True}


def simulate_pump(hamiltonian, t_extra=0, polarization='xx', pump=None,
                  rw_freq=None, basis='site', discard_imag_corr=False,
                  secular=True, subspace='01', append_thermal=False,
                  ode_settings=_ODE_DEFAULTS):
    """Simulate time evolution of the quantum state under a pump field"""
    if rw_freq is None:
        rw_freq = hamiltonian.system.central_freq
    hamiltonian = hamiltonian.to_rotating_frame(rw_freq)

    op_kwargs = dict(basis=basis, subspace=subspace)
    Lsys = CM_FS * redfield_evolve(hamiltonian, secular=secular,
                                   discard_imag_corr=discard_imag_corr,
                                   **op_kwargs)

    if pump is None:
        pump = GaussianPulse()

    V = [hamiltonian.dipole_destroy_evolve(polar, **op_kwargs)
         for polar in polarization]

    def drho(rho, t):
        # note: profiling shows 33% speedup when calculating the dot product
        # with rho before rather than after adding up different hamiltonian
        # terms
        return (Lsys.dot(rho)
                - pump(t, rw_freq) * V[0].dot(rho)
                - conj(pump(t, rw_freq)) * V[1].T.dot(rho))

    rho0 = unit_vec(0, hamiltonian.system.n_states(subspace) ** 2)
    t = np.arange(pump.t_init, pump.t_final + t_extra, hamiltonian.time_step)
    rho = odeint(drho, rho0, t, **ode_settings)

    if append_thermal:
        ground_pop = rho[-1, 0].real
        n_states = hamiltonian.system.n_states(subspace)
        rho_therm = np.zeros((n_states, n_states), dtype=complex)
        rho_therm[0, 0] = ground_pop
        rho_therm[1:, 1:] = (1 - ground_pop) * hamiltonian.thermal_state
        rho = np.append(rho, den_to_vec(rho_therm).reshape(1, -1), axis=0)
        t = np.append(t, np.infty)
    return MetaArray(rho, ticks=t, rw_freq=rw_freq, pump=pump)


def simulate_pump_(dynamics, pump=GaussianPulse(), polarization='xx',
                   t_extra=0):
    """Simulate time evolution of the quantum state under a pump field"""
    V = [dynamics.dipole_destroy(polarization[0]),
         dynamics.dipole_create(polarization[1])]

    def drho(rho, t):
        return (dynamics.step(rho)
                - pump(t) * V[0].step(rho)
                - conj(pump(t)) * V[1].step(rho))

    # rho0 = dynamics.ground_state
    t = np.arange(pump.t_init, pump.t_final + t_extra, dynamics.time_step)
    return dynamics.integrate(drho, t=t)


def simulate_dynamics_(dynamics, rho0, t_max=1000):
    """Simulate time evolution of the quantum state without any applied field"""
    def drho(rho, t):
        return dynamics.step(rho)
    rho0 = dynamics.from_density_vec(rho0)
    t = np.arange(0, t_max, dynamics.time_step)
    return dynamics.integrate(drho, rho0, t)


def simulate_dynamics(hamiltonian, rho0, t_max=1000, rw_freq=None, subspace='e',
                      basis='site', discard_imag_corr=False, secular=True,
                      sparse=False, ode_settings=_ODE_DEFAULTS):
    """Simulate time evolution of the quantum state without any applied field"""
    hamiltonian = hamiltonian.to_rotating_frame(rw_freq)
    Lsys = CM_FS * redfield_evolve(hamiltonian, secular=secular, basis=basis,
                                   discard_imag_corr=discard_imag_corr,
                                   subspace=subspace, sparse=sparse)

    def drho(rho, _):
        return Lsys.dot(rho)

    t = np.arange(0, t_max, hamiltonian.time_step)
    rho = odeint(drho, rho0, t, **ode_settings)
    return MetaArray(rho, ticks=t, rw_freq=rw_freq)


def simulate_hierarchy(hamiltonian, rho0, t_max=1000, rw_freq=None, tier_max=1,
                       dt=0.1, n_times=1000, debug=False):
    hamiltonian = hamiltonian.to_rotating_frame(rw_freq)
    t, rho = hierarchy.simulate(hamiltonian.system.H('e'),
                                hamiltonian.bath.temperature / CM_K,
                                hamiltonian.bath.reorg_energy,
                                hamiltonian.bath.cutoff_freq,
                                rho0, tier_max, t_max, dt, n_times, debug)
    return MetaArray(rho, ticks=t)


def validate_dynamics(subspace):
    def decorator(func):
        @wraps(func)
        def wrapper(dynamics, *args, **kwargs):
            if dynamics.subspace not in subspace:
                raise ValueError(
                    'invalid subspace {0}'.format(dynamics.subspace))
            return func(dynamics, *args, **kwargs)
        return wrapper
    return decorator


