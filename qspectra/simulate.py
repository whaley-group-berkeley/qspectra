from functools import wraps
import numpy as np

from .hamiltonian import optional_ensemble_average
from .polarization import optional_2nd_order_isotropic_average
from .utils import integrate, fourier_transform, Zero


@optional_ensemble_average
def simulate_dynamics(dynamical_model, initial_state, duration,
                      liouville_subspace='gg,ge,eg,ee', **integrate_kwargs):
    """
    Simulate time evolution with no applied field

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
        Object obeying the DynamicModel API.
    pulses : list of Pulse objects
        List of objects obeying the Pulse API.
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
                  liouville_subspace='gg,ge,eg,ee', isotropic_average=False,
                  **integrate_kwargs):
    """
    Simulate time evolution under a pump field in the rotating wave
    approximation

    Parameters
    ----------
    dynamical_model : DynamicalModel
        Object obeying the DynamicModel API.
    pump : Pulse
        Object obeying the Pulse API.
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
    isotropic_average : boolean, default False
        If True, perform an average over all molecular orientations (accurate
        up to 2nd order in the system-field coupling)

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
                isotropic_average=isotropic_average, **integrate_kwargs)


@optional_ensemble_average
@optional_2nd_order_isotropic_average
def linear_response(dynamical_model, liouv_space_path, time_max,
                    initial_state=None, polarization='xx', **integrate_kwargs):
    """
    Evaluate a linear response function

    Parameters
    ----------
    dynamical_model : DynamicalModel
        Object obeying the DynamicModel API.
    liouv_space_path : string of the form 'ab->cd->ef'
        String indicating the Liouville space pathways to include. Should
        indicate three valid Liouville subspaces separated by '->'.
    time_max : number
        Maximum time for which to simulate dynamics between the first and second
        interactions.
    initial_state : np.ndarray, optional
        Initial condition for the state vector, which should be defined on the
        first Liouville subspace in `liouv_space_path`. Defaults to the
        ground state of the dynamical model.
    polarization : pair of polarizations, default 'xx'
        Valid polarizations include:
        - 'x', 'y' or 'z', interepreted as the respective unit vectors
        - Angles of rotation from [1, 0, 0] in the x-y plane
        - 3D lists, tuples or arrays of numbers
    isotropic_average : boolean, default False
        If True, perform an average over all molecular orientations

    Returns
    -------
    t : np.ndarray
        Times at which the signal was simulated.
    signal : np.ndarray
        One-dimensional array containing the simulated complex valued electric
        field of the signal.
    """
    initial_liouv_subspace, int_liouv_subspace, final_liouv_subspace = \
        liouv_space_path.split('->')
    if initial_state is None:
        initial_state = dynamical_model.ground_state(initial_liouv_subspace)

    t = np.arange(0, time_max, dynamical_model.time_step)
    signal = Zero()
    for sim_subspace in int_liouv_subspace.split(','):
        V2 = dynamical_model.dipole_create(initial_liouv_subspace + '->'
                                           + sim_subspace, polarization[0])
        V2_rho = V2.commutator(initial_state)
        eom = dynamical_model.equation_of_motion(sim_subspace)
        V3 = dynamical_model.dipole_destroy(sim_subspace + '->'
                                            + final_liouv_subspace,
                                            polarization[1])
        signal -= integrate(eom, V2_rho, t, save_func=V3.expectation_value,
                            **integrate_kwargs)
    return (t, signal)


def return_fourier_transform(func):
    """
    Decorator that transforms the returned signal of this function from (t, x)
    to (f, X), where X is the Fourier transform of x.
    """
    @wraps(func)
    def wrapper(dynamical_model, *args, **kwargs):
        (t, x) = func(dynamical_model, *args, **kwargs)
        unit_convert = getattr(dynamical_model, 'unit_convert', None)
        (f, X) = fourier_transform(t, x, rw_freq=dynamical_model.rw_freq,
                                   unit_convert=unit_convert)
        return (f, X)
    return wrapper


def return_real_fourier_transform(func):
    """
    Decorator that transforms the returned signal of this function from (t, x)
    to (f, X.real), where X is the Fourier transform of x.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        (f, X) = return_fourier_transform(func)(*args, **kwargs)
        return (f, X.real)
    return wrapper


@return_real_fourier_transform
@optional_ensemble_average
@optional_2nd_order_isotropic_average
def absorption_spectra(dynamical_model, time_max, polarization='xx',
                       isotropic_average=False, **integrate_kwargs):
    """
    Returns the absorption spectra of a dynamical model

    Parameters
    ----------
    dynamical_model : DynamicalModel
        Object obeying the DynamicModel API.
    time_max : number
        Maximum time for which to simulate dynamics between the probe and signal
        interactions.
    polarization : iterable, default 'xx'
        Two item iterable giving the polarization of the last two system-field
        interactions as strings or 3D arrays
    isotropic_average : boolean, default False
        If True, perform an average over all molecular orientations

    Returns
    -------
    f : np.ndarray
        Frequencies at which the signal was simulated.
    signal : np.ndarray
        One-dimensional array containing the real valued absorption signal in
        the frequency domain.
    """
    (t, x) = linear_response(dynamical_model, 'gg->eg->gg', time_max,
                             polarization=polarization,
                             isotropic_average=isotropic_average,
                             **integrate_kwargs)
    return (t, -x)


PUMP_PROBE_PATHWAYS = {'GSB': 'gg->eg->gg',
                       'ESE': 'ee->eg->gg',
                       'ESA': 'ee->fe->ee'}


@return_fourier_transform
@optional_ensemble_average
@optional_2nd_order_isotropic_average
def impulsive_probe(dynamical_model, state, time_max, polarization='xx',
                    initial_liouv_subspace='gg,ge,eg,ee',
                    include_signal='GSB,ESE,ESA', **integrate_kwargs):
    """
    Probe the 2nd order portion of the provided state with an impulsive probe
    pulse under the rotating wave approximation

    Parameters
    ----------
    dynamical_model : DynamicalModel
        Object obeying the DynamicModel API.
    state : np.ndarray
        State vector for the system at the time of the probe pulse.
    time_max : number
        Maximum time for which to simulate dynamics between the probe and signal
        interactions.
    polarization : pair of polarizations, default 'xx'
        Valid polarizations include:
        - 'x', 'y' or 'z', interepreted as the respective unit vectors
        - Angles of rotation from [1, 0, 0] in the x-y plane
        - 3D lists, tuples or arrays of numbers
    initial_liouv_subspace : string, optional
        String indicating the subspace of Liouville space in which the provided
        state is defined. Defaults to 'gg,ge,eg,ee'.
    include_signal : container of any of 'GSB', 'ESE' and 'ESA'
        Indicates whether to include the ground-state-bleach (GSB), excited-
        state-emission (ESE) and excited-state-absorption (ESA) contributions
        to the signal.
    isotropic_average : boolean, default False
        If True, perform an average over all molecular orientations

    Returns
    -------
    f : np.ndarray
        Frequencies at which the signal was simulated.
    signal : np.ndarray
        One-dimensional array containing the simulated complex valued electric
        field of the signal in the frequency domain.
    """
    initial_state = state - dynamical_model.ground_state(initial_liouv_subspace)
    total_signal = Zero()
    for path in PUMP_PROBE_PATHWAYS:
        if path in include_signal:
            liouv_space_path = PUMP_PROBE_PATHWAYS[path]
            path_start = liouv_space_path.split('->')[0]
            init_state_portion = dynamical_model.map_between_subspaces(
                initial_state, initial_liouv_subspace, path_start)
            (t, signal) = linear_response(dynamical_model, liouv_space_path,
                                          time_max, init_state_portion,
                                          polarization, **integrate_kwargs)
            total_signal += signal
    if isinstance(total_signal, Zero):
        raise ValueError('include_signal must include at least one of '
                         "'GSB', 'ESE' or 'ESA'")
    return (t, total_signal)
