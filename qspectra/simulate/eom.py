"""
Equation of motion based methods non-linear spectra and dynamics
"""
import numpy as np

from .decorators import (optional_ensemble_average,
                         optional_2nd_order_isotropic_average)
from .utils import integrate
from ..operator_tools import basis_transform


@optional_ensemble_average
def _simulate_dynamics(dynamical_model, initial_state, duration, times,
                       liouville_subspace, save_func, **integrate_kwargs):
    eom = dynamical_model.equation_of_motion(liouville_subspace)
    t = (np.arange(0, duration, dynamical_model.time_step)
         if times is None else times)
    states = integrate(eom, initial_state, t, save_func=save_func,
                       **integrate_kwargs)
    return (t, states)


def simulate_dynamics(dynamical_model, initial_state, duration=None, times=None,
                      liouville_subspace='ee', save_func=None, basis='site',
                      ensemble_size=None, ensemble_random_orientations=False,
                      **integrate_kwargs):
    """
    Simulate time evolution with no applied field

    Parameters
    ----------
    dynamical_model : DynamicalModel
        Object obeying the DynamicModel interface.
    initial_state : np.ndarray
        Vector representing the initial state in Liouville space
    duration : number, optional
        Maximum time for which to simulate dynamics.
    times : array_like, optional
        Explicit times at which to return simulated dynamics. If provided,
        overrides duration.
    liouville_subspace : string, optional
        String indicating the subspace of Liouville space in which to integrate
        the equation of motion. Defaults to 'ee', the single excitation
        subspace.
    save_func : function, optional
        Optional function to apply to the state vector before returning it.
    basis : string, default 'site'
        Either 'site' or 'exciton'. specifies the basis of the initial condition
        and the output density matrices.
    ensemble_size : int, optional
        If provided, perform an ensemble average of this signal over Hamiltonian
        disorder, as determined by the `sample_ensemble` method of the provided
        dynamical model.
    ensemble_random_orientations : boolean, default False
        Whether or not to randomize the orientation of each member of the
        ensemble. Only relevant if `ensemble_size` is set.
    **integrate_kwargs : optional
        Keyword arguments passed on to `integrate`.

    Returns
    -------
    t : np.ndarray
        Times at which the state was simulated.
    states : np.ndarray
        Two-dimensional array of simulated state vectors at all times t.
    """

    U = dynamical_model.hamiltonian.U(liouville_subspace[0])

    if basis == 'site' and dynamical_model.evolve_basis == 'exciton':
        initial_state = basis_transform(initial_state, U)

    elif basis == 'exciton' and dynamical_model.evolve_basis == 'site':
        initial_state = basis_transform(initial_state, U.T.conj())

    t, states = _simulate_dynamics(
        dynamical_model, initial_state, duration, times, liouville_subspace,
        save_func, ensemble_size=ensemble_size,
        ensemble_random_orientations=ensemble_random_orientations,
        **integrate_kwargs)

    U = dynamical_model.hamiltonian.U(liouville_subspace[1])

    if basis == 'site' and dynamical_model.evolve_basis == 'exciton':
        for state in states:
            state[:] = basis_transform(state, U.T.conj())

    elif basis == 'exciton' and dynamical_model.evolve_basis == 'site':
        for state in states:
            state[:] = basis_transform(state, U)

    return t, states


# since optional_2nd_order_isotropic_average needs can only be applied *before*
# optional_ensemble_average, don't apply optional_ensemble_average yet
def _simulate_with_fields(dynamical_model, pulses, geometry, polarization,
                          time_extra, times, liouville_subspace, save_func,
                          **integrate_kwargs):
    eom = dynamical_model.equation_of_motion(liouville_subspace)
    V = [dynamical_model.dipole_operator(liouville_subspace, polar, trans)
         for polar, trans in zip(polarization, geometry)]
    field_info = zip(pulses, geometry, V)

    def f(t, state):
        deriv = eom(t, state)
        for pulse, trans, Vi in field_info:
            E = pulse(t, dynamical_model.rw_freq)
            if trans == '+':
                E = np.conj(E)
            deriv += (-1j * E) * Vi.commutator(state)
        return deriv

    initial_state = dynamical_model.thermal_state(liouville_subspace)

    t0 = min(p.t_init for p in pulses)
    tf = max(p.t_final for p in pulses)
    t = (np.arange(t0, tf + time_extra, dynamical_model.time_step)
         if times is None else tf + times)

    states = integrate(f, initial_state, t, t0=t0, save_func=save_func,
                       **integrate_kwargs)
    return (t, states)


def simulate_with_fields(dynamical_model, pulses, geometry='-+',
                         polarization='xx', time_extra=0, times=None,
                         liouville_subspace='gg,ge,eg,ee', save_func=None,
                         ensemble_size=None, ensemble_random_orientations=False,
                         **integrate_kwargs):
    """
    Simulate time evolution under a series of pulses in the rotating wave
    approximation

    Parameters
    ----------
    dynamical_model : DynamicalModel
        Object obeying the DynamicModel interface.
    pulses : list of Pulse objects
        List of objects obeying the Pulse interface.
    geometry : string
        String of '+' or '-' terms of the same length as pulses indicating
        whether to include a creation or annhilation operator with each pulse.
    polarization : list of polarizations
        List of polarizations of the same length as pulses. Valid polarizations
        include:
        - 'x', 'y' or 'z', interepreted as the respective unit vectors
        - Angles of rotation from [1, 0, 0] in the x-y plane
        - 3D lists, tuples or arrays of numbers
    time_extra : number, optional
        Extra time after the end of the pump-pulse for which to integrate
        dynamics (default 0).
    times : array_like, optional
        Explicit times relative to the end of the last pulse at which to return
        simulated dynamics. If provided, overrides time_extra.
    liouville_subspace : string, optional
        String indicating the subspace of Liouville space in which to integrate
        the equation of motion. Defaults to 'gg,ge,eg,ee', indicating all ground
        and single excitation states, an approximation which is valid for weak
        fields.
    save_func : function, optional
        Optional function to apply to the state vector before returning it.
    ensemble_size : int, optional
        If provided, perform an ensemble average of this signal over Hamiltonian
        disorder, as determined by the `sample_ensemble` method of the provided
        dynamical model.
    ensemble_random_orientations : boolean, default False
        Whether or not to randomize the orientation of each member of the
        ensemble. Only relevant if `ensemble_size` is set.

    Returns
    -------
    t : np.ndarray
        Times at which the state was simulated.
    states : np.ndarray
        Two-dimensional array of simulated state vectors at all times t.
    """
    return optional_ensemble_average(_simulate_with_fields)(
        dynamical_model, pulses, geometry, polarization, time_extra, times,
        liouville_subspace, save_func, ensemble_size=ensemble_size,
        ensemble_random_orientations=ensemble_random_orientations,
        **integrate_kwargs)


def simulate_pump(dynamical_model, pump, polarization='x', time_extra=0,
                  times=None, liouville_subspace='gg,ge,eg,ee', save_func=None,
                  ensemble_size=None, ensemble_random_orientations=False,
                  exact_isotropic_average=False, **integrate_kwargs):
    """
    Simulate time evolution under a pump field in the rotating wave
    approximation

    Parameters
    ----------
    dynamical_model : DynamicalModel
        Object obeying the DynamicModel interface.
    pump : Pulse
        Object obeying the Pulse interface.
    polarization : polarization, default 'x'
        Valid polarizations include:
        - 'x', 'y' or 'z', interepreted as the respective unit vectors
        - Angles of rotation from [1, 0, 0] in the x-y plane
        - 3D lists, tuples or arrays of numbers
    time_extra : number, optional
        Extra time after the end of the pump-pulse for which to integrate
        dynamics (default 0).
    times : array_like, optional
        Explicit times relative to the end of the last pulse at which to return
        simulated dynamics. If provided, overrides time_extra.
    liouville_subspace : string, optional
        String indicating the subspace of Liouville space in which to integrate
        the equation of motion. Defaults to 'gg,ge,eg,ee', indicating all ground
        and single excitation states, an approximation which is valid for weak
        fields.
    save_func : function, optional
        Optional function to apply to the state vector before returning it.
    ensemble_size : int, optional
        If provided, perform an ensemble average of this signal over Hamiltonian
        disorder, as determined by the `sample_ensemble` method of the provided
        dynamical model.
    ensemble_random_orientations : boolean, default False
        Whether or not to randomize the orientation of each member of the
        ensemble. Only relevant if `ensemble_size` is set.
    exact_isotropic_average : boolean, default False
        If True, perform an exact average over all molecular orientations
        (accurate up to 2nd order in the system-field coupling), at cost of 3x
        the computation time.
    **integrate_kwargs : optional
        Keyword arguments passed on to `integrate`.

    Returns
    -------
    t : np.ndarray
        Times at which the state was simulated.
    states : np.ndarray
        Two-dimensional array of simulated state vectors at all times t.
    """
    return optional_ensemble_average(
        optional_2nd_order_isotropic_average(_simulate_with_fields))(
            dynamical_model, [pump, pump], '-+', [polarization, polarization],
            time_extra, times, liouville_subspace, save_func,
            ensemble_size=ensemble_size,
            ensemble_random_orientations=ensemble_random_orientations,
            exact_isotropic_average=exact_isotropic_average,
            **integrate_kwargs)
