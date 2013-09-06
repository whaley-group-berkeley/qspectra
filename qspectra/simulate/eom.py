"""
Module for equation of motion based methods
"""

import numpy as np

from ..hamiltonian import optional_ensemble_average
from ..polarization import optional_2nd_order_isotropic_average
from .utils import integrate


@optional_ensemble_average
def simulate_dynamics(dynamical_model, initial_state, duration,
                      liouville_subspace='gg,ge,eg,ee', **integrate_kwargs):
    """
    Simulate time evolution with no applied field

    Parameters
    ----------
    dynamical_model : DynamicalModel
        Object obeying the DynamicModel interface.
    initial_state : np.ndarray
        Vector representing the initial state in Liouville space
    duration : number
        Time for which to simulate dynamics.
    liouville_subspace : string, optional
        String indicating the subspace of Liouville space in which to integrate
        the equation of motion. Defaults to 'gg,ge,eg,ee', indicating all ground
        and single excitation states, an approximation which is valid for weak
        fields.
    ensemble_size : int, optional
        If provided, perform an ensemble average of this signal over Hamiltonian
        disorder, as determined by the `sample_ensemble` method of the provided
        dynamical model.
    ensemble_random_orientations : boolean, default False
        Whether or not to randomize the orientation of each member of the
        ensemble. Only relevant if `ensemble_size` is set.
    ensemble_random_seed : int or array of int, optional
        Random seed for ensemble sampling.

    Returns
    -------
    t : np.ndarray
        Times at which the state was simulated.
    states : np.ndarray
        Two-dimensional array of simulated state vectors at all times t.
    """
    eom = dynamical_model.equation_of_motion(liouville_subspace)
    t = np.arange(0, duration, dynamical_model.time_step)
    states = integrate(eom, initial_state, t, **integrate_kwargs)
    return (t, states)


@optional_ensemble_average
def simulate_with_fields(dynamical_model, pulses, geometry='-+',
                         polarization='xx', time_extra=0,
                         liouville_subspace='gg,ge,eg,ee', **integrate_kwargs):
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
    liouville_subspace : string, optional
        String indicating the subspace of Liouville space in which to integrate
        the equation of motion. Defaults to 'gg,ge,eg,ee', indicating all ground
        and single excitation states, an approximation which is valid for weak
        fields.
    ensemble_size : int, optional
        If provided, perform an ensemble average of this signal over Hamiltonian
        disorder, as determined by the `sample_ensemble` method of the provided
        dynamical model.
    ensemble_random_orientations : boolean, default False
        Whether or not to randomize the orientation of each member of the
        ensemble. Only relevant if `ensemble_size` is set.
    ensemble_random_seed : int or array of int, optional
        Random seed for ensemble sampling.

    Returns
    -------
    t : np.ndarray
        Times at which the state was simulated.
    states : np.ndarray
        Two-dimensional array of simulated state vectors at all times t.
    """
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

    initial_state = dynamical_model.ground_state(liouville_subspace)
    t = np.arange(pulses[0].t_init, pulses[-1].t_final + time_extra,
                  dynamical_model.time_step)
    states = integrate(f, initial_state, t, **integrate_kwargs)
    return (t, states)


@optional_ensemble_average
def simulate_pump(dynamical_model, pump, polarization='x', time_extra=0,
                  liouville_subspace='gg,ge,eg,ee',
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
    liouville_subspace : string, optional
        String indicating the subspace of Liouville space in which to integrate
        the equation of motion. Defaults to 'gg,ge,eg,ee', indicating all ground
        and single excitation states, an approximation which is valid for weak
        fields.
    ensemble_size : int, optional
        If provided, perform an ensemble average of this signal over Hamiltonian
        disorder, as determined by the `sample_ensemble` method of the provided
        dynamical model.
    ensemble_random_orientations : boolean, default False
        Whether or not to randomize the orientation of each member of the
        ensemble. Only relevant if `ensemble_size` is set.
    ensemble_random_seed : int or array of int, optional
        Random seed for ensemble sampling.
    exact_isotropic_average : boolean, default False
        If True, perform an exact average over all molecular orientations
        (accurate up to 2nd order in the system-field coupling), at cost of 3x
        the computation time.
    **integrate_kwargs : optional
        Additional keyword arguments are passed to `utils.integrate`.

    Returns
    -------
    t : np.ndarray
        Times at which the state was simulated.
    states : np.ndarray
        Two-dimensional array of simulated state vectors at all times t.
    """
    return optional_2nd_order_isotropic_average(simulate_with_fields)(
                dynamical_model, [pump, pump], '-+',
                [polarization, polarization], time_extra, liouville_subspace,
                exact_isotropic_average=exact_isotropic_average,
                **integrate_kwargs)
