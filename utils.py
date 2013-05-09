import cPickle as pickle
import functools
import numpy as np
import scipy.integrate
import warnings


class Registry(dict):
    def __call__(self, obj):
        self[obj.__name__] = obj
        return obj


def mean(values):
    """Calculate the mean of all values from an iterable

    This function is convenient for taking the mean of an iterable of MetaArray
    objects while still keeping the metadata of the first object (like sum but
    unlike numpy.mean)
    """
    n = 0
    total = None
    for value in values:
        if total is None:
            total = value
        else:
            total += value
        n += 1
    return total / float(n)


class memoize(object):
    """
    Decorator that caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned, and
    not re-evaluated.
    Adapted from: http://wiki.python.org/moin/PythonDecoratorLibrary#Memoize
    """
    def __init__(self, func):
        self.func = func
        self.__doc__ = func.__doc__
        self.cache = {}

    def __call__(self, *args, **kwargs):
        key = (args, tuple(sorted(kwargs.iteritems())))
        try:
            return self.cache[key]
        except KeyError:
            value = self.func(*args, **kwargs)
            self.cache[key] = value
            return value
        except TypeError:
            warnings.warn(('memoize not able to cache key {0!r} for function '
                           '{1}').format(key, self.func))
            return self.func(*args, **kwargs)

    def __get__(self, obj, objtype):
        """Support instance methods."""
        return functools.partial(self.__call__, obj)


class imemoize(object):
    """cache the return value of a method

    This class is meant to be used as a decorator of methods. The return value
    from a given method invocation will be cached on the instance whose method
    was invoked. All arguments passed to a method decorated with memoize must
    be hashable.

    If a memoized method is invoked directly on its class the result will not
    be cached. Instead the method will be invoked like a static method:
    class Obj(object):
        @memoize
        def add_to(self, arg):
            return self + arg
    Obj.add_to(1) # not enough arguments
    Obj.add_to(1, 2) # returns 3, result is not cached
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


class memoize_by_pickle(object):
    """
    This version of memomize using the pickle of an object as the hash key,
    and hence memoize any function arguments. Calculating the pickle, however,
    adds several ms to every function call.
    """
    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args, **kwargs):
        haxh = pickle.dumps((args, sorted(kwargs.iteritems())))
        try:
            return self.cache[haxh]
        except KeyError:
            value = self.func(*args, **kwargs)
            self.cache[haxh] = value
            return value

    def __repr__(self):
        """Return the function's docstring."""
        return self.func.__doc__

    def __get__(self, obj, objtype):
        """Support instance methods."""
        return functools.partial(self.__call__, obj)


class MetaArray(np.ndarray):
    """
    Subclass of numpy.ndarray which stores metadata supplied as named arguments
    in a dictionary under the `metadata` attribute.

    Metadata keys can also be accessed directly as arrays attributes. Hence they
    are checked upon creating a MetaArray instance against pre-existing
    attributes to ensure there are no collisions.

    Example Usage
    -------------
    >>> import numpy as np
    >>> x = MetaArray(np.arange(5), extra='saved')
    >>> x.extra
    'saved'

    References
    ----------
    http://docs.scipy.org/doc/numpy/user/basics.subclassing.html
    """

    def __new__(cls, input_array, **metadata):
        """Create a new MetaArray object"""
        obj = np.asarray(input_array).view(cls)
        obj.metadata = metadata
        obj.check_metadata()
        return obj

    def __array_finalize__(self, obj):
        """Ensure a modified MetaArray is still a MetaArray"""
        if obj is not None:
            self.metadata = getattr(obj, 'metadata', {})

    def check_metadata(self):
        """Check to make sure metadata keys are not already attributes"""
        for key in self.metadata.iterkeys():
            if key in dir(self):
                raise AttributeError(
                    '{0} is already a named attribute'.format(key))

    def __getattr__(self, name):
        """Attribute style acccess for metadata"""
        try:
            return self.metadata[name]
        except KeyError:
            raise AttributeError('object has no attribute {0}'.format(name))

    def __repr__(self):
        metadata_str = ''.join(', {0}={1}'.format(k, v)
                               for k, v in self.metadata.iteritems())
        return '{0}({1}{2})'.format(type(self).__name__,
                                    np.ndarray.__repr__(self),
                                    metadata_str)

    def __reduce__(self):
        """Used by pickle"""
        state = np.ndarray.__reduce__(self)
        state[2] = (state[2], self.metadata)
        return tuple(state)

    def __setstate__(self, (array_state, metadata)):
        """Used by pickle"""
        np.ndarray.__setstate__(self, array_state)
        self.metadata = metadata


class IntegratorError(Exception):
    pass


def odeint(f, y0, t, method_name='zvode', f_params=None, save=None, load=None,
           return_meta=False, **kwdargs):
    """
    Functional interface to solvers from scipy.integrate.ode, providing
    syntax resembling scipy.integrate.odeint to solve the first-order
    differential equation:

        dy/dt = f(y, t, ...)

    with the initial value y0 at times specified by the vector t.

    **kwdargs are passed to set_integrator()

    To specify an integrator, method_name should be one of 'vode', 'zvode',
    'dopri5' or 'dop853' (the set of built in integrators), which will be passed
    on to set_integrator.

    Note that f takes arguments like f(y, t), the same order as
    scipy.integrate.odeint but opposite the order of scipy.integrate.ode.
    """
    if save is None:
        save = lambda y: y
    if load is None:
        load = lambda y: y

    shape = load(y0).shape

    def flatten(y):
        return y.reshape(-1)

    def unflatten(y):
        return y.reshape(shape)

    def g(t, y, **kwargs):
        return flatten(f(unflatten(y), t, **kwargs))

    solver = scipy.integrate.ode(g)
    solver.set_integrator(method_name, **kwdargs)
    solver.set_initial_value(flatten(load(y0)), t[0])
    if f_params is not None:
        solver.set_f_params(**f_params)

    y = np.empty([len(t)] + list(y0.shape), dtype=y0.dtype)
    y[0] = y0
    for i in xrange(1, len(t)):
        if solver.successful():
            y[i] = save(unflatten(solver.integrate(t[i])))
        else:
            raise IntegratorError('odeint ended early')
    return y if not return_meta else MetaArray(y, ticks=t)
