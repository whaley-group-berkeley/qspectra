"""
Module for response function based methods
"""
import numpy as np

from ..hamiltonian import optional_ensemble_average
from ..polarization import optional_2nd_order_isotropic_average
from .utils import (integrate, return_fourier_transform,
                    return_real_fourier_transform)
from ..utils import ZeroArray


@optional_ensemble_average
@optional_2nd_order_isotropic_average
def linear_response(dynamical_model, liouv_space_path, time_max,
                    initial_state=None, polarization='xx', **integrate_kwargs):
    """
    Evaluate a linear response function

    Parameters
    ----------
    dynamical_model : DynamicalModel
        Object obeying the DynamicModel interface.
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
        If True, perform an exact average over all molecular orientations, at
        cost of 3x the computation time.
    **integrate_kwargs : optional
        Additional keyword arguments are passed to `utils.integrate`.

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
    signal = ZeroArray()
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



@return_real_fourier_transform
@optional_ensemble_average
@optional_2nd_order_isotropic_average
def absorption_spectra(dynamical_model, time_max, correlation_decay_time=None,
                       polarization='xx', exact_isotropic_average=False,
                       **integrate_kwargs):
    """
    Returns the absorption spectra of a dynamical model

    Parameters
    ----------
    dynamical_model : DynamicalModel
        Object obeying the DynamicModel interface.
    time_max : number
        Maximum time for which to simulate dynamics between the probe and signal
        interactions.
    correlation_decay_time : number, optional
        If provided, multiply the dipole correlation function (i.e., the linear
        response function) by a exponential decay of the form `exp(-t/tau)`,
        where `tau` is this decay time.
    polarization : iterable, default 'xx'
        Two item iterable giving the polarization of the last two system-field
        interactions as strings or 3D arrays
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
        If True, perform an exact average over all molecular orientations, at
        cost of 3x the computation time.
    **integrate_kwargs : optional
        Additional keyword arguments are passed to `utils.integrate`.

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
                             exact_isotropic_average=exact_isotropic_average,
                             **integrate_kwargs)
    if correlation_decay_time is not None:
        x *= np.exp(-t / correlation_decay_time)
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
        Object obeying the DynamicModel interface.
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
        If True, perform an exact average over all molecular orientations, at
        cost of 3x the computation time.
    **integrate_kwargs : optional
        Additional keyword arguments are passed to `utils.integrate`.

    Returns
    -------
    f : np.ndarray
        Frequencies at which the signal was simulated.
    signal : np.ndarray
        One-dimensional array containing the simulated complex valued electric
        field of the signal in the frequency domain.
    """
    initial_state = state - dynamical_model.ground_state(initial_liouv_subspace)
    total_signal = ZeroArray()
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
    if isinstance(total_signal, ZeroArray):
        raise ValueError('include_signal must include at least one of '
                         "'GSB', 'ESE' or 'ESA'")
    return (t, total_signal)
