"""
Utility functions for producing and manipulating the output of simulations
"""
from numpy import pi
from scipy.fftpack import fft, fftshift, ifftshift, fftfreq
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


def is_constant(x, atol=1e-7, positive=None):
    x = np.asarray(x)
    return (np.max(np.abs(x - x[0])) < atol and
            (np.all((x > 0) == positive) if positive is not None else True))


def _symmetrize(t, x, axis=-1):
    if not is_constant(np.diff(t), positive=True):
        raise ValueError('sample times must differ by a positive constant')

    t = np.asarray(t)
    x = np.asarray(x)

    T = max(t[-1], -t[0])
    dt = t[1] - t[0]

    N_plus = int((T - t[-1]) / dt) + 1
    N_minus = int((T + t[0]) / dt) + 1

    t_sym = np.concatenate([t[0] - dt * np.arange(1, N_minus)[::-1], t,
                           t[-1] + dt * np.arange(1, N_plus)])

    new_shape = tuple(n if i != axis and i != x.ndim + axis
                      else t_sym.size
                      for i, n in enumerate(x.shape))
    x_sym = np.zeros(new_shape, dtype=x.dtype)
    start, end = (np.searchsorted(t_sym, ti) for ti in (t[0], t[-1]))
    x_sym[slice_along_axis(start, end + 1, axis=axis, ndim=x.ndim)] = x

    return t_sym, x_sym


def fourier_transform(t, x, axis=-1, rw_freq=0, unit_convert=1, sign=1,
                      convention='angular'):
    """
    Fourier transform a signal defined in a rotating frame using FFT

    By default, approximates the integral:
    .. math::
        X(\omega) = \int e^{i (\omega - \omega_0) t} x(t) dt

    where $\omega_0$ is the rotating wave frequency.

    The signal is assumed to be zero at any times at which it is not provided.

    Parameters
    ----------
    t : np.ndarray
        1D array giving the times at which the signal is defined.
    x : np.ndarray
        Signal to Fourier transform.
    axis : int, optional
        Axis along which to apply the Fourier transform to `x`.
    rw_freq : number, optional
        Frequency of the rotating frame in which the signal is sampled.
    unit_convert : number, optional
        Unit conversion from frequency to time units.
    sign : {1, -1}, optional
        Sign in the exponent.
    convention : {'angular', 'linear'}, optional
        Return angular or linear frequencies.

    Returns
    -------
    f : np.ndarray
        Frequencies at which the Fourier transformed signal is defined.
    X : np.ndarray
        The Fourier transformed signal.
    """
    if t.ndim != 1:
        raise ValueError('t must be one dimensional')
    if t.size != x.shape[axis]:
        raise ValueError('t must have the same length as the shape of x along '
                         'the given axis')
    if sign not in [-1, +1]:
        raise ValueError

    if convention == 'angular':
        unit_convert /= 2 * pi
    elif convention != 'linear':
        raise ValueError("convention must be 'angular' or 'linear'")

    t, x = _symmetrize(t, x, axis)

    N = x.shape[axis]
    dt = t[1] - t[0]

    f = fftshift(fftfreq(N, dt * unit_convert))
    X = fftshift(fft(ifftshift(x * dt, axes=axis), axis=axis), axes=axis)

    if sign == 1:
        f = -f[::-1]
        X = X[slice_along_axis(step=-1, axis=axis, ndim=X.ndim)]
    f += rw_freq
    return f, X


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
    nd_index = slice_along_axis(i0, i1 + 1, axis=axis, ndim=len(signal.shape))
    return ticks[i0:(i1 + 1)], signal[nd_index]
