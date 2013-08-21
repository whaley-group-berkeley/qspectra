from numpy import conj
import numpy as np

from utils import integrate


def simulate_pump(dynamical_model, pump, polarization, time_extra=0,
                  liouville_subspace='gg,ge,eg,ee', **integrate_kwargs):
    """
    Simulate time evolution under a pump field

    Parameters
    ----------
    dynamical_model : DynamicalModel
        Object obeying the DynamicModel API.
    pump : Pulse
        Object obeying the Pulse API.
    polarization : string or array
        String ('x', 'y' or 'z') or three item list giving the polarization of
        pump-field.
    time_extra : number, optional
        Extra time after the end of the pump-pulse for which to integrate
        dynamics (default 0).
    liouville_subspace : string, optional
        String indicating the subspace of Liouville space in which to integrate
        the equation of motion. Defaults to 'gg,ge,eg,ee', indicating all ground
        and single excitation states, an approximation which is valid for weak
        fields.

    Returns
    -------
    t : np.ndarray
        Times at which the state was simulated.
    states : np.ndarray
        Two-dimensional array of simulated state vectors at all times t.
    """
    eom = dynamical_model.equation_of_motion(liouville_subspace)
    V_minus = dynamical_model.dipole_destroy(liouville_subspace, polarization)
    V_plus = dynamical_model.dipole_create(liouville_subspace, polarization)

    def drho(t, rho):
        Et = pump(t, dynamical_model.rw_freq)
        return (eom(t, rho)
                + (-1j * Et) * V_minus.commutator(rho)
                + (-1j * conj(Et)) * V_plus.commutator(rho))

    initial_state = dynamical_model.ground_state(liouville_subspace)

    t = np.arange(pump.t_init, pump.t_final + time_extra,
                  dynamical_model.time_step)
    states = integrate(drho, initial_state, t, **integrate_kwargs)
    return (t, states)


def impulsive_probe(dynamical_model, rho_pump, time_max, polarization='xx',
                    rho_pump_liouv_subspace='gg,ge,eg,ee', **integrate_kwargs):
    """
    Probe an excited state with an impulsive probe pulse

    The signal includes the ground-state-bleach, excited-state-emission and
    excited-state-absorption contributions.

    Parameters
    ----------
    dynamical_model : DynamicalModel
        Object obeying the DynamicModel API.
    rho_pump : np.ndarray
        State vector for the system at the time of the probe pulse.
    time_max : number
        Maximum time for which to simulate dynamics between the probe and signal
        interactions.
    polarization : iterable
        Two item iterable giving the polarization of the last two system-field
        interactions as strings or 3D arrays
    rho_pump_liouv_subspace : string, optional
        String indicating the subspace of Liouville space in which rho_pump is
        defined. Defaults to 'gg,ge,eg,ee'.

    Returns
    -------
    t : np.ndarray
        Times at which the signal was simulated.
    signal : np.ndarray
        One-dimensional array containing the simulated complex valued electric
        field of the signal.
    """
    t = np.arange(0, time_max, dynamical_model.time_step)
    signal = np.zeros(t.shape, complex)
    rho2 = rho_pump - dynamical_model.ground_state(rho_pump_liouv_subspace)
    for sim_subspace in ['eg', 'fe']:
        V2 = dynamical_model.dipole_create(rho_pump_liouv_subspace + '->'
                                           + sim_subspace, polarization[0])
        V2_rho = V2.commutator(rho_pump)
        eom = dynamical_model.equation_of_motion(sim_subspace)
        V3 = dynamical_model.dipole_destroy(sim_subspace + '->gg,ee',
                                            polarization[1])
        # the minus sign comes from factors of i from the response function
        # and electric field definitions
        signal -= integrate(eom, V2_rho, t, save_func=V3.expectation_value,
                            **integrate_kwargs)
        return (t, signal)


def simulate_dynamics(dynamical_model, initial_state, duration,
                      liouville_subspace='gg,ge,eg,ee', **integrate_kwargs):
    """
    Simulate time evolution under a pump field

    Parameters
    ----------
    dynamical_model : DynamicalModel
        Object obeying the DynamicModel API.
    initial_state : np.ndarray
        Vector representing the initial state in Liouville space
    duration : number
        Time for which to simulate dynamics.
    liouville_subspace : string, optional
        String indicating the subspace of Liouville space in which to integrate
        the equation of motion. Defaults to 'gg,ge,eg,ee', indicating all ground
        and single excitation states, an approximation which is valid for weak
        fields.

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

