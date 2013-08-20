from numpy import conj
import numpy as np

from utils import integrate


def simulate_pump(dynamical_model, pump, polarization, time_extra=0,
                  subspace='g,e'):
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
    subspace : string
        String indicating the subspace in which to integrate the equation of
        motion. Defaults to 'g,e' indicating all ground and single excitation
        states, an approximation which is valid for weak fields.

    Returns
    -------
    t : np.ndarray
        Times at which the state was simulated.
    states : np.ndarray
        Two-dimensional array of simulated state vectors at all times t.
    """

    eom = dynamical_model.equation_of_motion(subspace)
    V_minus = dynamical_model.dipole_destroy(subspace, polarization)
    V_plus = dynamical_model.dipole_create(subspace, polarization)

    def drho(t, rho):
        Et = pump(t, dynamical_model.rw_freq)
        return (eom(rho)
                + (-1j * Et) * V_minus.commutator(rho)
                + (-1j * conj(Et)) * V_plus.commutator(rho))

    rho0 = dynamical_model.ground_state(subspace)
    t = np.arange(pump.t_init, pump.t_final + time_extra,
                  dynamical_model.time_step)
    states = integrate(drho, rho0, t)
    return (t, states)
