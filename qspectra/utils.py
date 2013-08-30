from copy import copy
from numpy import pi
import functools
import numpy as np
import scipy.integrate


class IntegratorError(Exception):
    pass


def integrate(f, y0, t, method_name='zvode', f_params=None, save_func=None,
              **kwargs):
    """
    Functional interface to solvers from scipy.integrate.ode, providing
    syntax resembling scipy.integrate.odeint to solve the first-order
    differential equation:

        dy/dt = f(t, y, ...)

    with the initial value y0 at times specified by the vector t.

    Parameters
    ----------
    f : function
        Funtion to integrate. Should take arguments like f(t, y, **f_params).
    y0 : np.ndarray
        Initial value.
    t : np.ndarray
        Times at which to return the calculate state of the system. The system
        is assumed to be in the state y0 at time t[0].
    method_name : string, optional
        Method name to pass to scipy.integrate.ode (default 'zvode').
    f_params : dict, optional
        Additional parameters to call f with.
    save_func : function, optional
        Function to call on a state y to select the desired return values. By
        default, the entire state vector is returned.
    **kwargs : optional
        Additional arguments to pass to the set_integrator of the
        scipy.integrate.ode instance used to solve this ODE.

    Returns
    -------
    y : np.ndarray, shape (len(t), len(save_func(y0)))
        2D array containing the results of calling save_func on the state of the
        integrator at all given times t.
    """
    if save_func is None:
        save_func = lambda x: x

    solver = scipy.integrate.ode(f)
    solver.set_integrator(method_name, **kwargs)
    solver.set_initial_value(y0, t[0])
    if f_params is not None:
        solver.set_f_params(**f_params)

    y0_saved = save_func(y0)
    try:
        save_shape = list(y0_saved.shape)
    except AttributeError:
        # y0_saved is just a number
        save_shape = []
    y = np.empty([len(t)] + save_shape, dtype=y0_saved.dtype)

    y[0] = y0_saved
    for i in xrange(1, len(t)):
        if solver.successful():
            y[i] = save_func(solver.integrate(t[i]))
        else:
            raise IntegratorError('integration failed at time {}'.format(t[i]))
    return y


class Zero(object):
    """
    Zero is an abstract representation of the number zero

    After an inplace addition (+=) or subtraction (-=), a Zero object is
    replaced by the other object (or its negative).

    The purpose of Zero is to allow for initializing arrays of unknown
    dimensions and type.

    It's useful to initialize arrays of all zeros on which to perform inplace
    operations, of course, because it allows for inplace array summations inside
    a `for` loop, which can be much faster (and simplier) than calling Python's
    built-in `sum` (which is not inplace).

    Example
    -------
    >>> x = Zero()
    >>> x += 10
    >>> x
    10
    >>> type(x)
    int
    """
    def __iadd__(self, other):
        return other

    def __isub__(self, other):
        return -other


def fourier_transform(t, x, axis=0, rw_freq=0, unit_convert=1, freq_bounds=None,
                      reverse_freq=False, positive_time_only=True):
    """
    Fourier transform a signal defined in a rotating frame using FFT

    By default, approximates the integral:
        X(\omega) = \int e^{i \omega t} x(t)

    Parameters
    ----------
    t : np.ndarray
        Times at which the signal is defined.
    x : np.ndarray
        Signal to Fourier transform.
    axes : int, optional
        Axis along which to apply the Fourier transform (default 0).
    rw_freq : number, optional
        Frequency of the rotating frame in which the signal is sampled.
    unit_convert : number, optional
        Unit conversion from frequency to time units (default 1).
    freq_bounds : list of 2 numbers, optional
        Bounds giving the minimum and maximum frequencies at which to return
        the transformed signal.
    reverse_freq : boolean, optional
        Switch the exponential from +\omega to -\omega.
    positive_time_only : boolean, optional
        If True (default), the signal is assumed to only be defined for at
        positive times, and the signal is zero-padded on the left with len(x)
        zeros before passing it to the FFT routine.
    """
    if positive_time_only:
        x_all = np.concatenate([np.zeros_like(x), x], axis=axis)
    else:
        x_all = x
    x_shifted = np.fft.fftshift(x_all, axes=axis)

    X = np.fft.fftshift(np.fft.fft(x_shifted, axis=axis), axes=axis)
    dt = t[1] - t[0]

    rev = 1 if reverse_freq else -1

    fft_freqs = np.fft.fftfreq(x_all.shape[axis], dt * unit_convert / (2 * pi))
    freqs = rev * np.fft.fftshift(fft_freqs, axes=axis) + rw_freq

    if freq_bounds is not None:
        i0 = np.argmin(np.abs(freqs - freq_bounds[0]))
        i1 = np.argmin(np.abs(freqs - freq_bounds[1]))
        X = X[min(i0, i1):(max(i0, i1) + 1)]
        freqs = freqs[min(i0, i1):(max(i0, i1) + 1)]

    return freqs[::rev], X[::rev]


class imemoize(object):
    """
    Cache the return value of a method

    This class is meant to be used as a decorator of methods. The return value
    from a given method invocation will be cached on the instance whose method
    was invoked. All arguments passed to the decorated method decorated must be
    be hashable.

    Source (MIT Licensed)
    --------------------
        http://code.activestate.com/recipes/
        577452-a-memoize-decorator-for-instance-methods/
    """
    def __init__(self, func):
        self.func = func

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self.func
        return functools.partial(self, obj)

    def __call__(self, *args, **kw):
        obj = args[0]
        try:
            cache = obj.__cache
        except AttributeError:
            cache = obj.__cache = {}
        key = (self.func, args[1:], frozenset(kw.items()))
        try:
            res = cache[key]
        except KeyError:
            res = cache[key] = self.func(*args, **kw)
        return res


def copy_with_new_cache(obj):
    """
    Return a shallow copy of the provided object, resetting the cache as used by
    imemoize (if present)
    """
    new_obj = copy(obj)
    try:
        del new_obj._imemoize__cache
    except AttributeError:
        pass
    return new_obj


def memoized_property(x):
    return property(imemoize(x))
