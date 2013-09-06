from copy import copy
import functools


class ZeroArray(object):
    """
    ZeroArray is an abstract representation of the number zero

    After an inplace addition (+=) or subtraction (-=), a ZeroArray object is
    replaced by the other object (or its negative).

    The purpose of ZeroArray is to allow for initializing arrays of unknown
    dimensions and type.

    It's useful to initialize arrays of all zeros on which to perform inplace
    operations, of course, because it allows for inplace array summations inside
    a `for` loop, which can be much faster (and simplier) than calling Python's
    built-in `sum` (which is not inplace).

    Example
    -------
    >>> x = ZeroArray()
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
