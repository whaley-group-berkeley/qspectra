"""
Response function based methods for calculating linear and non-linear spectra
"""
import numpy as np

from .decorators import (optional_ensemble_average,
                         optional_2nd_order_isotropic_average,
                         optional_4th_order_isotropic_average)
from .utils import integrate, fourier_transform
from ..utils import ZeroArray


@optional_ensemble_average
@optional_2nd_order_isotropic_average
def _linear_response(dynamical_model, liouv_space_path, time_max,
                     initial_state=None, polarization='xx', **integrate_kwargs):
    subspaces = liouv_space_path.split('->')
    if initial_state is None:
        initial_state = dynamical_model.ground_state(subspaces[0])

    t = np.arange(0, time_max, dynamical_model.time_step)
    signal = ZeroArray()
    for sim_subspace in subspaces[1].split(','):
        V = [dynamical_model.dipole_operator(
                '{}->{}'.format(sub_start, sub_end), polar, trans)
             for sub_start, sub_end, polar, trans
             in zip(subspaces[:-1], subspaces[1:], polarization, '+-')]
        V_rho2 = np.apply_along_axis(V[0].commutator, -1, initial_state)
        try:
            # attempt to integrate using the Heisenberg picture, since it is
            # much faster if there is more than one initial_state 
            eom = dynamical_model.equation_of_motion(sim_subspace,
                                                     heisenberg_picture=True)
        except NotImplementedError:
            # fall back on the Schroedinger picture
            eom = dynamical_model.equation_of_motion(sim_subspace)
            signal -= integrate(eom, V_rho2, t,
                                save_func=V[1].expectation_value,
                                **integrate_kwargs)
        else:
            V_Gt3 = integrate(eom, -V[1].bra_vector, t, **integrate_kwargs)
            signal += np.tensordot(V_rho2, V_Gt3, (-1, -1))
    return (t, signal)


def linear_response(dynamical_model, liouv_space_path, time_max,
                    initial_state=None, polarization='xx', ensemble_size=None,
                    ensemble_random_orientations=False,
                    exact_isotropic_average=False, **integrate_kwargs):
    """
    Evaluate a linear response function under the rotating wave approximation

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
        Initial condition(s) for the state vector, which should be defined on
        the first Liouville subspace in `liouv_space_path`. Defaults to the
        ground state of the dynamical model.
    polarization : pair of polarizations, optional
        Valid polarizations include:
        - 'x', 'y' or 'z', interepreted as the respective unit vectors
        - Angles of rotation from [1, 0, 0] in the x-y plane
        - 3D lists, tuples or arrays of numbers
    ensemble_size : int, optional
        If provided, perform an ensemble average of this signal over Hamiltonian
        disorder, as determined by the `sample_ensemble` method of the provided
        dynamical model.
    ensemble_random_orientations : boolean, optional
        Whether or not to randomize the orientation of each member of the
        ensemble. Only relevant if `ensemble_size` is set.
    exact_isotropic_average : boolean, optional
        If True, perform an exact average over all molecular orientations, at
        cost of 3x the computation time.
    **integrate_kwargs : optional
        Keyword arguments passed on to `integrate`.

    Returns
    -------
    t : np.ndarray
        Times at which the signal was simulated.
    signal : np.ndarray
        One-dimensional array containing the simulated complex valued electric
        field of the signal.
    """
    return _linear_response(
        dynamical_model, liouv_space_path, time_max, initial_state,
        polarization, ensemble_size=ensemble_size,
        ensemble_random_orientations=ensemble_random_orientations,
        exact_isotropic_average=exact_isotropic_average,
        **integrate_kwargs)


def absorption_spectra(dynamical_model, time_max, correlation_decay_time=None,
                       polarization='xx', ensemble_size=None,
                       ensemble_random_orientations=False,
                       exact_isotropic_average=False, **integrate_kwargs):
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
    polarization : iterable, optional
        Two item iterable giving the polarization of the last two system-field
        interactions as strings or 3D arrays
    ensemble_size : int, optional
        If provided, perform an ensemble average of this signal over Hamiltonian
        disorder, as determined by the `sample_ensemble` method of the provided
        dynamical model.
    ensemble_random_orientations : boolean, optional
        Whether or not to randomize the orientation of each member of the
        ensemble. Only relevant if `ensemble_size` is set.
    exact_isotropic_average : boolean, optional
        If True, perform an exact average over all molecular orientations, at
        cost of 3x the computation time.
    **integrate_kwargs : optional
        Keyword arguments passed on to `integrate`.

    Returns
    -------
    f : np.ndarray
        Frequencies at which the signal was simulated.
    signal : np.ndarray
        One-dimensional array containing the real valued absorption signal in
        the frequency domain.
    """
    (t, x) = linear_response(
        dynamical_model, 'gg->eg->gg', time_max, polarization=polarization,
        ensemble_size=ensemble_size,
        ensemble_random_orientations=ensemble_random_orientations,
        exact_isotropic_average=exact_isotropic_average, **integrate_kwargs)
    if correlation_decay_time is not None:
        x *= np.exp(-t / correlation_decay_time)
    (f, X) = fourier_transform(t, -x, rw_freq=dynamical_model.rw_freq,
                               unit_convert=dynamical_model.unit_convert)
    return (f, X.real)


PUMP_PROBE_PATHWAYS = {'GSB': 'gg->eg->gg',
                       'ESE': 'ee->eg->gg',
                       'ESA': 'ee->fe->ee'}


def _parse_pathways(possible_pathways, include_signal):
    selected_pathways = []
    for k, v in possible_pathways.iteritems():
        if include_signal is None or k in include_signal:
            selected_pathways.append(v)
    if not selected_pathways:
        raise ValueError('at least one Liouville space pathway must be '
                         'selected, i.e., include_signal must include at least '
                         'one of %r' % possible_pathways.keys())
    return selected_pathways


def impulsive_probe(dynamical_model, state, time_max, polarization='xx',
                     initial_liouv_subspace='gg,ge,eg,ee',
                     include_signal='GSB,ESE,ESA', ensemble_size=None,
                     ensemble_random_orientations=False,
                     exact_isotropic_average=False, **integrate_kwargs):
    """
    Probe the 2nd order portion of the provided state with an impulsive probe
    pulse under the rotating wave approximation

    Parameters
    ----------
    dynamical_model : DynamicalModel
        Object obeying the DynamicModel interface.
    state : np.ndarray
        State vector(s) for the system at the time of the probe pulse.
    time_max : number
        Maximum time for which to simulate dynamics between the probe and signal
        interactions.
    polarization : pair of polarizations, optional
        Valid polarizations include:
        - 'x', 'y' or 'z', interepreted as the respective unit vectors
        - Angles of rotation from [1, 0, 0] in the x-y plane
        - 3D lists, tuples or arrays of numbers
    initial_liouv_subspace : string, optional
        String indicating the subspace of Liouville space in which the provided
        state is defined.
    include_signal : container of any of 'GSB', 'ESE' and 'ESA', optional
        Indicates whether to include the ground-state-bleach (GSB), excited-
        state-emission (ESE) and excited-state-absorption (ESA) contributions
        to the signal.
    ensemble_size : int, optional
        If provided, perform an ensemble average of this signal over Hamiltonian
        disorder, as determined by the `sample_ensemble` method of the provided
        dynamical model.
    ensemble_random_orientations : boolean, optional
        Whether or not to randomize the orientation of each member of the
        ensemble. Only relevant if `ensemble_size` is set.
    exact_isotropic_average : boolean, optional
        If True, perform an exact average over all molecular orientations, at
        cost of 3x the computation time.
    **integrate_kwargs : optional
        Keyword arguments passed on to `integrate`.

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

    selected_pathways = _parse_pathways(PUMP_PROBE_PATHWAYS, include_signal)
    for liouv_space_path in selected_pathways:
        def map_subspace(state):
            return dynamical_model.map_between_subspaces(
                state, initial_liouv_subspace, liouv_space_path.split('->')[0])
        init_state_portion = np.apply_along_axis(map_subspace, -1, initial_state)
        (t, signal) = linear_response(
            dynamical_model, liouv_space_path, time_max, init_state_portion,
            polarization, ensemble_size=ensemble_size,
            ensemble_random_orientations=ensemble_random_orientations,
            exact_isotropic_average=exact_isotropic_average, **integrate_kwargs)
        total_signal += signal
    return fourier_transform(t, total_signal, rw_freq=dynamical_model.rw_freq,
                             unit_convert=dynamical_model.unit_convert)


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
def _third_order_response(dynamical_model, coherence_time_max,
                          population_time_max, population_times, geometry,
                          polarization, include_signal, **integrate_kwargs):
    # This is a reasonable first draft. However, there are definitely some
    # significant possible improvements in computational efficiency:
    # (1) We use the Heisenberg picture to avoid expensive integration loops
    #     only during the third time interval. Could we always use the
    #     Heisenberg picture instead of the Schroedinger picture? In principle,
    #     this could result in speedups of ~50x, since we would no longer need
    #     to loop over different times t1.
    # (2) Instead of using the `commutator` method of each dipole operator, we
    #     could use the `left_multiply` or `right_multiply` methods, based on
    #     whether the dipole operator is of creation or annihilation type (as
    #     determined by `geometry`) and whether the change in the density matrix
    #     is on the left or right sides (as indicated in
    #     `THIRD_ORDER_PATHWAYS`). Every right multplication would require a
    #     matching multiplication by -1. This book-keeping, however, would not
    #     save us any time in the integration steps (which are the probably the
    #     most expensive part of the calculation). But something like this might
    #     be necessary to implement (1), in which case it would be worth it.
    # (3) There are some redundant calculations, because some of the Liouville
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

    selected_pathways = _parse_pathways(THIRD_ORDER_PATHWAYS[geometry],
                                        include_signal)
    for liouv_space_path in selected_pathways:
        subspaces = liouv_space_path.split('->')
        V = [dynamical_model.dipole_operator(
                '{}->{}'.format(sub_start, sub_end), polar, trans)
             for sub_start, sub_end, polar, trans
             in zip(subspaces[:-1], subspaces[1:], polarization,
                    geometry + '-')]
        eom = [dynamical_model.equation_of_motion(subspace)
               for subspace in subspaces[1:-1]]
        V_rho0 = V[0].commutator(initial_state)
        V_rho1 = integrate(eom[0], V_rho0, t1, save_func=V[1].commutator,
                           **integrate_kwargs)
        V_rho2 = integrate(eom[1], V_rho1, t2, t0=0, save_func=V[2].commutator,
                           **integrate_kwargs)
        try:
            # attempt to integrate over t3 using the Heisenberg picture,
            # since it requires far less computational effort
            eom_heisen = dynamical_model.equation_of_motion(
                subspaces[3], heisenberg_picture=True)
        except NotImplementedError:
            # fall back on using the Schroedinger picture
            total_signal += integrate(eom[2], V_rho2, t3,
                                      save_func=V[3].expectation_value,
                                      **integrate_kwargs)
        else:
            V_Gt3 = integrate(eom_heisen, V[3].bra_vector, t3,
                              **integrate_kwargs)
            total_signal += np.einsum('ci,abi', V_Gt3, V_rho2)
    return (t1, t2, t3), total_signal


def third_order_response(dynamical_model, coherence_time_max,
                         population_time_max=None, population_times=None,
                         geometry='-++', polarization='xxxx',
                         include_signal=None, ensemble_size=None,
                         ensemble_random_orientations=False,
                         exact_isotropic_average=False, **integrate_kwargs):
    """
    Evaluate a third order response function in the rotating wave approximation

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
        Explicit times at which to simulate dynamics between the second and
        third interactions. If provided, overrides population_time_max.
    geometry : '-++', '+-+' or '++-', optional
        String of '+' or '-' terms indicating whether to simulate the so-called
        photon-echo signal $-k_1 + k_2 + k_3$ ('-++'), the non-rephasing signal
        $k_1 - k_2 + k_3$ ('+-+') or the double-quantum-coherence signal
        $k_1 + k_2 - k_3$ ('++-').
    polarization : four polarizations, optional
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
    ensemble_random_orientations : boolean, optional
        Whether or not to randomize the orientation of each member of the
        ensemble. Only relevant if `ensemble_size` is set.
    exact_isotropic_average : boolean, optional
        If True, perform an exact average over all molecular orientations, at
        cost of 3x the computation time.
    **integrate_kwargs : optional
        Keyword arguments passed on to `integrate`.

    Returns
    -------
    (t1, t2, t3) : tuple of np.ndarray
        Coherence, population and rephasing times at which the signal was
        simulated.
    signal : np.ndarray
        3D array of shape (len(t1), len(t2), len(t3)) containing the simulated
        complex valued electric field of the signal.

    Note
    ----
    This function calculates the third order response function by explicitly
    summing over all possible Liouville space pathways [1]. It also uses the
    trick of using the Heisenberg picture to integrate the equations of motion
    during the last time interval, which speeds up calculations by ~100x over
    the naive loop over all values of t1 and t2 [2].

    References
    ----------
    [1] Abramavicius, D., Palmieri, B., Voronine, D. V., Sanda, F. & Mukamel, S.
        Coherent Multidimensional Optical Spectroscopy of Excitons in Molecular
        Aggregates; Quasiparticle versus Supermolecule Perspectives. Chem. Rev.
        109, 2350-2408 (2009).
    [2] Xu, J., Xu, R.-X., Abramavicius, D., Zhang, H. & Yan, Y. Advancing
        hierarchical equations of motion for efficient evaluation of
        coherent two-dimensional spectroscopy. Chin. J. Chem. Phys 24, 497
        (2011). arXiv:1109.6168
    """
    return _third_order_response(
        dynamical_model, coherence_time_max, population_time_max,
        population_times, geometry, polarization, include_signal,
        ensemble_size=ensemble_size,
        ensemble_random_orientations=ensemble_random_orientations,
        exact_isotropic_average=exact_isotropic_average,
        **integrate_kwargs)
