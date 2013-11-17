"""
Decorator functions to facilitate writing simulation methods
"""
from functools import wraps
import inspect

import numpy as np

from ..polarization import (check_polarizations, invariant_weights_4th_order,
                            invariant_polarizations, FOURTH_ORDER_INVARIANTS)
from ..utils import ZeroArray


def _get_call_args(func, *args, **kwargs):
    """
    Like inspect.getcallargs, except it returns keyword arguments inserted
    directly into the return dictionary

    The idea behind this variant is that the original function can safely be
    called like `func(**call_args)` -- except that this dictionary of call
    arguments could potentially be modified (e.g., by a function decorator).

    I.e., the following should to be equivalent:
    >>> func(*args, **kwargs)
    >>> func(**_get_call_args(func, *args, **kwargs))

    Accordingly, if `func` is defined with positional arguments, `get_call_args`
    raises a NotImplementedError.
    """
    spec = inspect.getargspec(func)
    if spec.varargs is not None:
        raise NotImplementedError(('%s cannot include positional-only '
                                   'arguments (i.e., of the form *args)')
                                  % func)
    call_args = inspect.getcallargs(func, *args, **kwargs)
    call_args.update(call_args.pop(spec.keywords, {}))
    return call_args


def optional_ensemble_average(func):
    """
    Function decorator to add optional `ensemble_size` and
    `ensemble_random_orientations` keyword arguments to a function that takes a
    dynamical model as its first argument

    If `ensemble_size` is set, the function is resampled that number of times
    with dynamical models yielded by the original dynamical model's
    `sample_ensemble` method.
    """
    @wraps(func)
    def wrapper(dynamical_model, *args, **kwargs):
        ensemble_size = kwargs.pop('ensemble_size', None)
        random_orientations = kwargs.pop('ensemble_random_orientations', False)
        if ensemble_size is not None:
            total_signal = ZeroArray()
            for dyn_model in dynamical_model.sample_ensemble(
                    ensemble_size, random_orientations):
                (ticks, signal) = func(dyn_model, *args, **kwargs)
                total_signal += signal
            total_signal /= ensemble_size
            return (ticks, total_signal)
        else:
            return func(dynamical_model, *args, **kwargs)
    return wrapper


def optional_4th_order_isotropic_average(func):
    """
    Function decorator to add an optional `exact_isotropic_average` argument to
    a function with a `polarization` argument that takes four values

    If `exact_isotropic_average` is True, the function is instead called
    repeatedly in order to calculate the exact isotropic average of this signal
    for the given lab frame polarization.

    The calculation is done by calculating all necessary 4th order tensor
    variants of the function, and thus requires between 9 and 21 total function
    evaluations.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if kwargs.pop('exact_isotropic_average', False):
            kwargs = _get_call_args(func, *args, **kwargs)
            weights = invariant_weights_4th_order(kwargs.pop('polarization'))
            signals = {}
            total_signal = ZeroArray()
            for invariant, weight in zip(FOURTH_ORDER_INVARIANTS, weights):
                if weight > 1e-8:
                    for p in invariant_polarizations(invariant):
                        if p not in signals:
                            (t, signals[p]) = func(polarization=p, **kwargs)
                        total_signal += weight * signals[p]
            return (t, total_signal)
        else:
            return func(*args, **kwargs)
    return wrapper


def optional_2nd_order_isotropic_average(func):
    """
    Function decorator to add an optional `exact_isotropic_average` argument to
    a function with a `polarization` argument that takes a pair of values

    If `exact_isotropic_average` is True, the function is instead called
    repeatedly in order to calculate the exact isotropic average of this signal
    for the given lab frame polarization.

    The calculation is done by calculating the 2nd order tensor variant of the
    function, which requires 3 total function evaluations.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if kwargs.pop('exact_isotropic_average', False):
            kwargs = _get_call_args(func, *args, **kwargs)
            polarizations = check_polarizations(kwargs.pop('polarization'), 2)
            weight = np.dot(*polarizations)
            total_signal = ZeroArray()
            for p in ['xx', 'yy', 'zz']:
                (t, signal) = func(polarization=p, **kwargs)
                total_signal += signal
            total_signal *= weight / 3.0
            return (t, total_signal)
        else:
            return func(*args, **kwargs)
    return wrapper
