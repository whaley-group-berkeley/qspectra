from numpy import conj
import numpy as np

# TODO (SH): define the integrate function (probably by adapting a simple
# wrapper of ZVODE) and figure out how to pass it options
from somewhere import integrate


def simulate_pump(dynamics, pump, polarization, time_extra=0, subspace='g,e'):
    """
    Simulate time evolution under a pump field

    Parameters
    ----------
    dynamics : dynamics.Dynamics
        Object obeying the Dynamics API.
    pump : pulse.Pulse instance
        Object obeying the Pulse API.
    polarization : string or array
        String ('x', 'y' or 'z') or three item list giving the polarization of
        pump-field.
    time_extra : number, optional
        Extra time after the end of the pump-pulse for which to integrate
        dynamics (default 0).
    subspace : string
        String indicating the subspace in which to calculate dynamics. Defaults
        to 'g,e' indicating all ground and single excitation states, an
        approximation which is valid for weak fields.

    Returns
    -------
    t : np.ndarray
        Times at which the state was simulated
    states : np.ndarray
        Two-dimensional array of simulated state vectors at all times t
    """

    eom = dynamics.equation_of_motion(subspace)
    V_minus = dynamics.dipole_destroy(polarization, subspace)
    V_plus = dynamics.dipole_create(polarization, subspace)

    def drho(rho, t):
        Et = pump(t, dynamics.rw_freq)
        return (eom(rho)
                + Et * V_minus.negative_imag_commutator(rho)
                + conj(Et) * V_plus.negative_imag_commutator(rho))

    rho0 = dynamics.ground_state(subspace)
    t = np.arange(pump.t_init, pump.t_final + time_extra, dynamics.time_step)
    states = integrate(drho, rho0, t)
    return (t, states)
