from itertools import product
from numbers import Number
from numpy import cos, sin, pi, sqrt
import numpy as np

from .utils import check_random_state


COORD_MAP = {'x': np.array([1, 0, 0]),
             'y': np.array([0, 1, 0]),
             'z': np.array([0, 0, 1])}


def polarization_vector(p):
    """
    Cast a polarization into a 3D vector of floats

    Valid polarizations include:

    - Strings 'x', 'y' or 'z', interepreted as the respective unit vectors
    - Single numbers, interpreted as angles of rotation from [1, 0, 0] in the
      x-y plane
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
                raise ValueError('polarization vectors must have length 3')
            return p
    except:
        raise ValueError('invalid polarization {}'.format(p))


def check_polarizations(p, length):
    polarizations = np.array(map(polarization_vector, p))
    if len(polarizations) != length:
        raise ValueError('%s polarizations required' % length)
    return polarizations


FOURTH_ORDER_INVARIANTS = [((0, 1), (2, 3)),
                           ((0, 2), (1, 3)),
                           ((0, 3), (1, 2))]


MAGIC_ANGLE = np.arccos(1 / np.sqrt(3))
r"""The Magic Angle :math:`\approx 54.7\degree` in radians"""


def invariant_weights_4th_order(polarizations):
    """
    Given four polarizations in the lab frame, return the weights on each of the
    three 4th order tensor invariants <xxyy>, <xyxy> and <xyyx>, with
    which to sum the invariants in order to determine the isotropic average
    for this set of polarizations

    References
    ----------
    .. [1] Hochstrasser, R. M. Two-dimensional IR-spectroscopy: polarization
       anisotropy effects. Chem. Phys. 266, 273-284 (2001).
    """
    polarizations = check_polarizations(polarizations, 4)
    cosines = np.einsum('ni,mi->nm', polarizations, polarizations)
    products = [cosines[x] * cosines[y] for x, y in FOURTH_ORDER_INVARIANTS]
    return (5 * np.eye(3) - np.ones((3, 3))).dot(products) / 30


def invariant_polarizations(invariant):
    """
    Given a 4th order tensor invariant, written in terms of pairs of Kronecker
    delta functions (i.e., an element of FOURTH_ORDER_INVARIANTS), returns a
    list of the 9 polarization configurations over which to sum 3rd order
    signals in order to calculate this invariant
    """
    if invariant not in FOURTH_ORDER_INVARIANTS:
        raise ValueError('`invariant` is not one of the three 4th order '
                         'tensor invariants %r' % FOURTH_ORDER_INVARIANTS)
    return [''.join(polarization) for polarization in product('xyz', repeat=4)
            if all(polarization[a] == polarization[b] for a, b in invariant)]


def random_rotation_matrix(random_state=None):
    """
    Returns a uniformly distributed random rotation matrix

    Reference
    ---------
    .. [1] Arvo, J. Fast Random Rotation Matrices in Graphics Gems III 117-120
       (Academic Press Professional, Inc., 1992).
    """
    x1, x2, x3 = check_random_state(random_state).rand(3)
    theta = 2 * pi * x1
    phi = 2 * pi * x2
    R = [[cos(theta), sin(theta), 0], [-sin(theta), cos(theta), 0], [0, 0, 1]]
    v = [cos(phi) * sqrt(x3), sin(phi) * sqrt(x3), sqrt(1 - x3)]
    H = np.identity(3) - 2 * np.outer(v, v)
    return -H.dot(R)
