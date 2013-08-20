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
        return (eom(rho)
                + (-1j * Et) * V_minus.commutator(rho)
                + (-1j * conj(Et)) * V_plus.commutator(rho))

    initial_state = dynamical_model.ground_state(liouville_subspace)

    t = np.arange(pump.t_init, pump.t_final + time_extra,
                  dynamical_model.time_step)
    states = integrate(drho, initial_state, t, **integrate_kwargs)
    return (t, states)


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

    def drho(t, rho):
        return eom(rho)

    t = np.arange(0, duration, dynamical_model.time_step)
    states = integrate(drho, initial_state, t, **integrate_kwargs)
    return (t, states)
