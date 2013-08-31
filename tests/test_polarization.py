import numpy as np
import unittest
from numpy import pi
from numpy.testing import assert_equal, assert_allclose

from qspectra import polarization


class TestPolarization(unittest.TestCase):
    def test_polarization_vector(self):
        assert_equal(polarization.polarization_vector('x'),
                     [1, 0, 0])
        assert_equal(polarization.polarization_vector(0),
                     [1, 0, 0])
        assert_allclose(polarization.polarization_vector(pi / 2),
                        [0, 1, 0], atol=1e-10)
        with self.assertRaises(polarization.PolarizationError):
            polarization.polarization_vector('X')
        with self.assertRaises(polarization.PolarizationError):
            polarization.polarization_vector([1, 0])

    def test_invariant_weights_4th_order(self):
        # Hochstrasser, Eq. 9(a):
        assert_allclose(polarization.invariant_weights_4th_order('xxxx'),
                        np.ones(3) / 15)
        # Hochstrasser, Eq. 9(b):
        assert_allclose(polarization.invariant_weights_4th_order('xxyy'),
                        np.array([4, -1, -1]) / 30.0)
        # magic angle:
        ma = polarization.MAGIC_ANGLE
        assert_allclose(
            polarization.invariant_weights_4th_order([0, 0, ma, ma]),
            [1 / 9.0, 0, 0], atol=1e-10)
        # see:
        # Zanni, M. T., Ge, N. H., Kim, Y. S. & Hochstrasser, R. M. Two-
        # dimensional IR spectroscopy can be designed to eliminate the diagonal
        # peaks and expose only the crosspeaks needed for structure
        # determination. Proc. Natl Acad. Sci. USA 98, 11265-11270 (2001).
        assert_allclose(
            polarization.invariant_weights_4th_order(
                [pi / 4, -pi / 4, pi / 2, 0]),
            [0, 1 / 12.0, -1 / 12.0], atol=1e-10)
        # consistency checks:
        assert_allclose(polarization.invariant_weights_4th_order('xxxx'),
                        polarization.invariant_weights_4th_order('yyyy'))
        assert_allclose(polarization.invariant_weights_4th_order('xxxx'),
                        polarization.invariant_weights_4th_order('zzzz'))
        assert_allclose(polarization.invariant_weights_4th_order('xxyy'),
                        polarization.invariant_weights_4th_order('yyxx'))
        assert_allclose(polarization.invariant_weights_4th_order('xxyy'),
                        polarization.invariant_weights_4th_order('zzyy'))

    def test_list_polarizations(self):
        with self.assertRaises(polarization.PolarizationError):
            polarization.list_polarizations(((0, 1), (0, 2)))
        invariants = polarization.FOURTH_ORDER_INVARIANTS
        for invariant in invariants:
            self.assertEqual(len(polarization.list_polarizations(invariant)),
                             9)
        self.assertItemsEqual(polarization.list_polarizations(invariants[0]),
                              ['xxxx', 'xxyy', 'xxzz',
                               'yyxx', 'yyyy', 'yyzz',
                               'zzxx', 'zzyy', 'zzzz'])


class TestGetCallArgs(unittest.TestCase):
    def test(self):
        self.assertEqual(
            polarization._get_call_args(lambda a: None, 1),
            {'a': 1})
        self.assertEqual(
            polarization._get_call_args(lambda a, **b: None, 1),
            {'a': 1})
        self.assertEqual(
            polarization._get_call_args(lambda a, **b: None, a=1, c=2),
            {'a': 1, 'c': 2})
        self.assertEqual(
            polarization._get_call_args(lambda **b: None, a=1, c=2),
            {'a': 1, 'c': 2})
        with self.assertRaises(NotImplementedError):
            polarization._get_call_args(lambda *a: None, 1, 2, 3)


class TestIsotropicAverage(unittest.TestCase):
    def test_optional_2nd_order_isotropic_average(self):
        binary = {'xx': 1, 'yy': 2, 'zz': 4}
        f = polarization.optional_2nd_order_isotropic_average(
            lambda polarization: (0, binary[polarization]))
        assert_allclose(f('xx'), (0, 1))
        assert_allclose(f('xx', isotropic_average=False), (0, 1))
        assert_allclose(f('xx', isotropic_average=True), (0, 7 / 3.0))
        assert_allclose(f('xy', isotropic_average=True), (0, 0))

    def test_optional_4th_order_isotropic_average(self):
        binary = {'xx': 1, 'yy': 2, 'zz': 4}
        f = polarization.optional_4th_order_isotropic_average(
            lambda polarization: (0, binary[polarization[:2]]
                                      + 10 * binary[polarization[2:]]))
        assert_allclose(f('xxxx'), (0, 11))
        ma = polarization.MAGIC_ANGLE
        assert_allclose(f([0, 0, ma, ma], isotropic_average=True),
                        (0, (11 + 12 + 14 + 21 + 22 + 24 + 41 + 42 + 44) / 9.0))
