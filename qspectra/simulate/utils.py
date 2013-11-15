"""
Utility functions for producing and manipulating the output of simulations
"""
from functools import wraps

from numpy import pi
import numpy as np
import scipy.integrate

from ..utils import ndarray_list


class IntegratorError(Exception):
    pass


def _integrate(f, y0, t, t0, method_name, f_params, save_func, **kwargs):
    if t0 is None:
        t0 = t[0]
    if method_name == 'zvode':
        # this is the method to ode that supports complex values 
        solver = scipy.integrate.ode(f)
    else:
        solver = scipy.integrate.complex_ode(f)
    solver.set_integrator(method_name, **kwargs)
    solver.set_initial_value(y0, t0)
    if f_params is not None:
        solver.set_f_params(**f_params)

    if save_func is None:
        save_func = lambda x: x
    y0_saved = save_func(y0)
    try:
        save_shape = list(y0_saved.shape)
    except AttributeError:
        # in this case, y0_saved is just a number
        save_shape = []
    y = np.empty([len(t)] + save_shape, dtype=y0_saved.dtype)

    if t[0] == t0:
        # integrate.ode does not like being asked for the initial time
        y[0] = y0_saved
        i0 = 1
    else:
        i0 = 0
    for i in xrange(i0, len(t)):
        if solver.successful():
            y[i] = save_func(solver.integrate(t[i]))
        else:
            raise IntegratorError('integration failed at time {}'.format(t[i]))
    return y


def integrate(f, y0, t, t0=None, method_name='zvode', f_params=None,
              save_func=None, **kwargs):
    """
    Functional interface to solvers from scipy.integrate.ode, providing
    syntax resembling scipy.integrate.odeint to solve the first-order
    differential equation:

        dy/dt = f(t, y, ...)

    with the initial value y0 at times specified by the vector t.

    If y0 is a higher than 1-dimensional vector, integrate only integrates over
    the last axes, looping over all prior axes.

    Parameters
    ----------
    f : function
        Funtion to integrate. Should take arguments like f(t, y, **f_params).
    y0 : np.ndarray
        Initial value at time t0.
    t : np.ndarray
        Times at which to return the state of the system.
    t0 : float, optional
        Time at which to start the integration. Defaults to t[0].
    method_name : string, optional
        Method name to pass to scipy.integrate.ode (default 'zvode'). If
        method_name is not 'zvode', scipy.integrate.complex_ode is used instead.
    f_params : dict, optional
        Additional parameters to pass to `f`.
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
    if len(y0.shape) == 1:
        return _integrate(f, y0, t, t0, method_name, f_params, save_func,
                          **kwargs)
    else:
        return ndarray_list((integrate(f, y0i, t, t0, method_name, f_params,
                                       save_func, **kwargs)
                             for y0i in y0), len(y0))


def slice_along_axis(start=None, stop=None, step=None, axis=0, ndim=1):
    """
    Returns an N-dimensional slice along only the specified axis
    """
    return tuple(slice(start, stop, step)
                 if (n == axis) or (n == ndim + axis)
                 else slice(None)
                 for n in xrange(ndim))


def fourier_transform(t, x, axis=-1, rw_freq=0, unit_convert=1,
                      reverse_freq=False, positive_time_only=True):
    """
    Fourier transform a signal defined in a rotating frame using FFT

    By default, approximates the integral:
        X(\omega) = \int e^{i \omega t} x(t)

    Parameters
    ----------
    t : np.ndarray
        1D array giving the times at which the signal is defined.
    x : np.ndarray
        Signal to Fourier transform.
    axis : int, optional
        Axis along which to apply the Fourier transform to `x` (defaults to -1).
    rw_freq : number, optional
        Frequency of the rotating frame in which the signal is sampled.
    unit_convert : number, optional
        Unit conversion from frequency to time units (default 1).
    reverse_freq : boolean, optional
        Switch the sign in the exponent from + to -.
    positive_time_only : boolean, optional
        If True (default), the signal is assumed to only be defined for at
        positive times, and the signal is zero-padded on the left with len(x)
        zeros before passing it to the FFT routine.

    Returns
    -------
    f : np.ndarray
        Frequencies at which the Fourier transformed signal is defined.
    X : np.ndarray
        The Fourier transformed signal.
    """
    # TODO: update this function so it doesn't need a `positive_time_only`
    # switch but can enlarge x and t to the right size automatically
    if len(np.unique(np.diff(t))) != 1:
        raise ValueError('Sample times must differ by a constant')
    if len(t.shape) > 1:
        raise ValueError('t must be one dimensional')
    if len(t) != x.shape[axis]:
        raise ValueError('t must have the same length as the shape of x along '
                         'the given axis')

    x_all = (np.concatenate([np.zeros_like(x), x], axis=axis)
             if positive_time_only else x)
    x_shifted = np.fft.fftshift(x_all, axes=axis)

    X = np.fft.fftshift(np.fft.fft(x_shifted, axis=axis), axes=axis)
    dt = t[1] - t[0]

    fft_freqs = np.fft.fftfreq(x_all.shape[axis], dt * unit_convert / (2 * pi))
    rev = 1 if reverse_freq else -1
    freqs = rev * np.fft.fftshift(fft_freqs) + rw_freq

    nd_index = slice_along_axis(step=rev, axis=axis, ndim=len(X.shape))
    return freqs[::rev], X[nd_index]


def bound_signal(ticks, signal, bounds, axis=0):
    """
    Bound a signal by tick values along a given axis

    Parameters
    ----------
    ticks : np.ndarray
        1D array giving the tick marks for the signal
    signal : np.ndarray
        ND array giving the signal to bound
    bounds : list of 2 numbers
        Bounds giving the minimum and maximum ticks at which to return the
        signal
    axis : int, default 0
        Axis along which to bound signal

    Returns
    -------
    ticks : np.ndarray
        Bounded ticks
    signal : np.ndarray
        Bounded signal
    """
    if signal.shape[axis] != len(ticks):
        raise ValueError('ticks must have same shape as signal along given '
                         'axis')
    i0, i1 = sorted(np.argmin(np.abs(ticks - bound)) for bound in bounds)
    nd_index = slice_along_axis(i0, i1 + 1, None, axis, len(signal.shape))
    return ticks[i0:(i1 + 1)], signal[nd_index]


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
