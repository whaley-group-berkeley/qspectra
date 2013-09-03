from functools import wraps
from itertools import product
from numbers import Number
from numpy import cos, sin, pi, sqrt
import inspect
import numpy as np

from .utils import Zero


COORD_MAP = {'x': np.array([1, 0, 0]),
             'y': np.array([0, 1, 0]),
             'z': np.array([0, 0, 1])}


class PolarizationError(Exception):
    pass


def polarization_vector(p):
    """
    Cast a polarization into a 3D vector of floats

    Valid polarizations include:
    - 'x', 'y' or 'z', interepreted as the respective unit vectors
    - Angles of rotation from [1, 0, 0] in the x-y plane
    - 3D lists, tuples or arrays of numbers
    """
    try:
        if isinstance(p, str):
            return COORD_MAP[p]
        elif isinstance(p, Number):
            return np.array([np.cos(p), np.sin(p), 0])
        else:
            p = np.asanyarray(p, float).reshape(-1)
            if len(p) != 3:
                raise PolarizationError('Polarization vectors must have length 3')
            return p
    except:
        raise PolarizationError('invalid polarization {}'.format(p))


FOURTH_ORDER_INVARIANTS = (((0, 1), (2, 3)),
                           ((0, 2), (1, 3)),
                           ((0, 3), (1, 2)))


MAGIC_ANGLE = np.arccos(1 / np.sqrt(3))


def invariant_weights_4th_order(polarizations):
    """
    Given four polarizations in the lab frame, return the weights on each of the
    three 4th order tensor invariants <xxyy>, <xyxy> and <xyyx>, with
    which to sum the invariants in order to determine the isotropic average
    for this set of polarizations

    Reference
    ---------
    Hochstrasser, R. M. Two-dimensional IR-spectroscopy: polarization anisotropy
    effects. Chem. Phys. 266, 273-284 (2001).
    """
    polarizations = map(polarization_vector, polarizations)
    cosines = np.einsum('ni,mi->nm', polarizations, polarizations)
    products = [cosines[x] * cosines[y] for x, y in FOURTH_ORDER_INVARIANTS]
    return (5 * np.eye(3) - np.ones((3, 3))).dot(products) / 30


def list_polarizations(invariant):
    """
    Given a 4th order tensor invariant, written in terms of pairs of Kronecker
    delta functions (i.e., an element of FOURTH_ORDER_INVARIANTS), returns a
    list of the 9 polarization configurations over which to sum 3rd order
    signals in order to calculate this invariant
    """
    if invariant not in FOURTH_ORDER_INVARIANTS:
        raise PolarizationError('`invariant` is not one of the three 4th order '
                                'tensor invariants')
    return [''.join(polarization) for polarization in product('xyz', repeat=4)
            if all(polarization[a] == polarization[b] for a, b in invariant)]


def random_rotation_matrix(random_seed=None):
    """
    Returns a uniformly distributed random rotation matrix

    Reference
    ---------
    Arvo, J. Fast Random Rotation Matrices in Graphics Gems III 117-120
    (Academic Press Professional, Inc., 1992).
    """
    np.random.seed(random_seed)
    x1, x2, x3 = np.random.rand(3)
    theta = 2 * pi * x1
    phi = 2 * pi * x2
    R = [[cos(theta), sin(theta), 0], [-sin(theta), cos(theta), 0], [0, 0, 1]]
    v = [cos(phi) * sqrt(x3), sin(phi) * sqrt(x3), sqrt(1 - x3)]
    H = np.identity(3) - 2 * np.outer(v, v)
    return -H.dot(R)


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
        raise NotImplementedError(('{} cannot includes positional arguments '
                                   '(i.e., of the form *args)').format(func))
    call_args = inspect.getcallargs(func, *args, **kwargs)
    call_args.update(call_args.pop(spec.keywords, {}))
    return call_args


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
            total_signal = Zero()
            for invariant, weight in zip(FOURTH_ORDER_INVARIANTS, weights):
                if weight > 1e-8:
                    for p in list_polarizations(invariant):
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
            weight = np.dot(*map(polarization_vector,
                                 kwargs.pop('polarization')))
            total_signal = Zero()
            for p in ['xx', 'yy', 'zz']:
                (t, signal) = func(polarization=p, **kwargs)
                total_signal += signal
            total_signal *= weight / 3.0
            return (t, total_signal)
        else:
            return func(*args, **kwargs)
    return wrapper
