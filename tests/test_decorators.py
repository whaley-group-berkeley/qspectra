import unittest
from numpy.testing import assert_allclose

from qspectra import polarization
from qspectra.simulate import decorators


class TestGetCallArgs(unittest.TestCase):
    def test(self):
        self.assertEqual(
            decorators._get_call_args(lambda a: None, 1),
            {'a': 1})
        self.assertEqual(
            decorators._get_call_args(lambda a, **b: None, 1),
            {'a': 1})
        self.assertEqual(
            decorators._get_call_args(lambda a, **b: None, a=1, c=2),
            {'a': 1, 'c': 2})
        self.assertEqual(
            decorators._get_call_args(lambda **b: None, a=1, c=2),
            {'a': 1, 'c': 2})
        with self.assertRaises(NotImplementedError):
            decorators._get_call_args(lambda *a: None, 1, 2, 3)


class TestIsotropicAverage(unittest.TestCase):
    def test_optional_2nd_order_isotropic_average(self):
        binary = {'xx': 1, 'yy': 2, 'zz': 4}
        f = decorators.optional_2nd_order_isotropic_average(
            lambda polarization: (0, binary[polarization]))
        assert_allclose(f('xx'), (0, 1))
        assert_allclose(f('xx', exact_isotropic_average=False), (0, 1))
        assert_allclose(f('xx', exact_isotropic_average=True), (0, 7 / 3.0))
        assert_allclose(f('xy', exact_isotropic_average=True), (0, 0))
        with self.assertRaises(ValueError):
            # wrong number of polarizations
            f('xyz', exact_isotropic_average=True)

    def test_optional_4th_order_isotropic_average(self):
        binary = {'xx': 1, 'yy': 2, 'zz': 4}
        f = decorators.optional_4th_order_isotropic_average(
            lambda polarization: (0, binary[polarization[:2]]
                                      + 10 * binary[polarization[2:]]))
        assert_allclose(f('xxxx'), (0, 11))
        ma = polarization.MAGIC_ANGLE
        assert_allclose(f([0, 0, ma, ma], exact_isotropic_average=True),
                        (0, (11 + 12 + 14 + 21 + 22 + 24 + 41 + 42 + 44) / 9.0))
        with self.assertRaises(ValueError):
            # wrong number of polarizations
            f('xyz', exact_isotropic_average=True)
