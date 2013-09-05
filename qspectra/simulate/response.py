"""
Module for response function based methods
"""
import numpy as np

from ..hamiltonian import optional_ensemble_average
from ..polarization import (optional_2nd_order_isotropic_average,
                            optional_4th_order_isotropic_average)
from .utils import (integrate, return_fourier_transform,
                    return_real_fourier_transform)
from ..utils import ZeroArray, ndarray_list


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
    geometry : string
        String of '+' or '-' terms of the same length as pulses indicating
        whether to include a creation or annhilation operator with each pulse.
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


# Copied from Figures 4, 5 and 6 of:
# Abramavicius, D., Palmieri, B., Voronine, D. V., Sanda, F. & Mukamel, S.
# Coherent Multidimensional Optical Spectroscopy of Excitons in Molecular
# Aggregates; Quasiparticle versus Supermolecule Perspectives. Chem. Rev. 109,
# 2350-2408 (2009).
THIRD_ORDER_PATHWAYS = {
    # photon-echo
    '-++': {'ESE': 'gg->ge->ee->eg->gg',
            'GSB': 'gg->ge->gg->eg->gg',
            'ESA': 'gg->ge->ee->fe->ee'},
    # non-rephasing
    '+-+': {'ESE': 'gg->eg->ee->eg->gg',
            'GSB': 'gg->eg->gg->eg->gg',
            'ESA': 'gg->eg->ee->fe->ee'},
    # double-quantum-coherence
    '++-': {'ESA1': 'gg->eg->fg->fe->ee',
            'ESA2': 'gg->eg->fg->eg->gg'}
}


@optional_ensemble_average
@optional_4th_order_isotropic_average
def third_order_response(dynamical_model, coherence_time_max,
                         population_time_max=None, population_times=None,
                         geometry='-++', polarization='xxxx',
                         include_signal=None, **integrate_kwargs):
    """
    Evaluate a linear response function

    Parameters
    ----------
    dynamical_model : DynamicalModel
        Object obeying the DynamicModel interface.
    coherence_time_max : number
        Maximum time for which to simulate dynamics between the first and second
        and betwen the third and fourth interactions (i.e., for the rephasing
        time and as well as the coherence time).
    population_time_max : number, optional
        Maximum time for which to simulate dynamics between the second and third
        interactions.
    population_times : number, optional
        Explicit times at which to simulate dynamics between the second and third
        interactions. If provided, overrides population_time_max.
    geometry : '-++' (default), '+-+' or '++-'
        String of '+' or '-' terms indicating whether to simulate the so-called
        photon-echo signal $-k_1 + k_2 + k_3$ ('-++'), the non-rephasing signal
        $k_1 - k_2 + k_3$ ('+-+') or the double-quantum-coherence signal
        $k_1 + k_2 - k_3$ ('++-').
    polarization : four polarizations, default 'xxxx'
        Valid polarizations include:
        - 'x', 'y' or 'z', interepreted as the respective unit vectors
        - Angles of rotation from [1, 0, 0] in the x-y plane
        - 3D lists, tuples or arrays of numbers
    include_signal : container of any of 'GSB', 'ESE' and 'ESA', optional
        Indicates whether to include the ground-state-bleach (GSB), excited-
        state-emission (ESE) and excited-state-absorption (ESA) contributions to
        the signal. In the double-quantum-coherence geometry (++-), valid
        choices are 'ESA1' and 'ESA2'. By default all pathways are included.
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
    (t1, t2, t3) : tuple of np.ndarray
        Coherence, population and rephasing times at which the signal was
        simulated.
    signal : np.ndarray
        One-dimensional array containing the simulated complex valued electric
        field of the signal.
    """
    # This is a reasonable first draft. However, there could be two significant
    # improvements for the next version:
    # (1) Instead of using the `commutator` method of each dipole operator, we
    #     could use the `left_multiply` or `right_multiply` methods, based on
    #     whether the dipole operator is of creation or annihilation type (as
    #     determined by `geometry`) and whether the change in the density matrix
    #     is on the left or right sides (as indicated in
    #     `THIRD_ORDER_PATHWAYS`). Every right multplication would require a
    #     matching multiplication by -1. This book-keeping, however, would not
    #     save us anytime in the integration steps (which are the probably the
    #     most expensive part of the calculation).
    # (2) There are some redudant calculations, because some of the Liouville
    #     space pathways are equivalent up to or after certain interactions. For
    #     example, all photon-echo pathways start with 'gg->ge', and both GSB
    #     and ESE photon-echo pathways end with 'eg->gg'.

    t1 = np.arange(0, coherence_time_max, dynamical_model.time_step)
    if population_times is None:
        t2 = np.arange(0, population_time_max, dynamical_model.time_step)
    else:
        t2 = population_times
    t3 = np.arange(0, coherence_time_max, dynamical_model.time_step)

    initial_state = dynamical_model.ground_state('gg')
    total_signal = ZeroArray()
    for path in THIRD_ORDER_PATHWAYS[geometry]:
        if include_signal is None or path in include_signal:
            subspaces = THIRD_ORDER_PATHWAYS[geometry][path].split('->')
            V = [dynamical_model.dipole_operator(
                    '{}->{}'.format(sub_start, sub_end), polar, trans)
                 for sub_start, sub_end, polar, trans
                 in zip(subspaces[:-1], subspaces[1:], polarization,
                        geometry + '-')]
            eom = [dynamical_model.equation_of_motion(subspace)
                   for subspace in subspaces[1:-1]]
            V_rho0 = V[0].commutator(initial_state)
            V_rho1 = integrate(eom[0], V_rho0, t1,
                               save_func=V[1].commutator,
                                **integrate_kwargs)
            V_rho2 = ndarray_list(
                         (integrate(eom[1], init, t2,
                                   save_func=V[2].commutator,
                                   **integrate_kwargs)
                          for init in V_rho1),
                         len(V_rho1))
            total_signal += ndarray_list(
                                (ndarray_list(
                                    (integrate(eom[2], init_inner, t3,
                                               save_func=V[3].expectation_value,
                                               **integrate_kwargs)
                                     for init_inner in init_outer),
                                    len(init_outer))
                                 for init_outer in V_rho2),
                                len(V_rho2))
    if isinstance(total_signal, ZeroArray):
        if geometry == '++-':
            raise ValueError('include_signal must include at least one of '
                             "'ESA1' or 'ESA2'")
        else:
            raise ValueError('include_signal must include at least one of '
                             "'GSB', 'ESE' or 'ESA'")
    return (t1, t2, t3), total_signal