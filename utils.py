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

    y = np.empty([len(t)] + list(save_func(y0).shape), dtype=y0.dtype)
    y[0] = save_func(y0)
    for i in xrange(1, len(t)):
        if solver.successful():
            y[i] = save_func(solver.integrate(t[i]))
        else:
            raise IntegratorError('integration failed at time {}'.format(t[i]))
    return y


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


def memoized_property(x):
    return property(imemoize(x))
