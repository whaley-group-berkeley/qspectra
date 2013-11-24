import numpy as np
import unittest
from numpy.testing import assert_allclose

from qspectra.simulate import utils


class FourierTransform(object):
    def verify(self, t):
        v, X = utils.fourier_transform(t, self.f(t))
        self.assertTrue(utils.is_constant(np.diff(v), positive=True))
        assert_allclose(X, self.F(v), atol=self.atol)

        v, X = utils.fourier_transform(t, self.f(t), sign=-1)
        self.assertTrue(utils.is_constant(np.diff(v), positive=True))
        assert_allclose(X, self.F(-v), atol=self.atol)

        v, X = utils.fourier_transform(t, self.f(t), rw_freq=5)
        self.assertTrue(utils.is_constant(np.diff(v), positive=True))
        assert_allclose(X, self.F(v - 5), atol=self.atol)

    def test_balanced(self):
        self.verify(np.linspace(-1000, 1000, num=10000))

    def test_unbalanced(self):
        self.verify(np.linspace(-1500, 1000, num=10000))

    def test_odd(self):
        self.verify(np.linspace(-1500, 1000, num=10001))

# Examples courtesy of:
# http://en.wikipedia.org/wiki/Fourier_transform#Square-integrable_functions

class TestPositiveExponentialFT(unittest.TestCase, FourierTransform):
    def setUp(self):
        a = (1.0 + 0.3j) / 100
        self.f = lambda x: (x > 0) * np.exp(-a * x)
        self.F = lambda v: 1.0 / (a - 1j * v)
        self.atol = 0.002 * np.abs(self.F(0))

    def test_positive_only(self):
        self.verify(np.linspace(0, 1000, num=10000))

    def test_positive_mostly(self):
        self.verify(np.linspace(-50, 1000, num=10000))


class TestGaussianFT(unittest.TestCase, FourierTransform):
    def setUp(self):
        a = 1.0 / 100
        b = 0.3 / 100
        self.f = lambda x: np.exp(-a * x ** 2 - 1j * b * x)
        self.F = lambda v: np.sqrt(np.pi / a) * np.exp(-(v - b) ** 2 / (4 * a))
        self.atol = 0.01 * np.abs(self.F(0))


def test_is_constant():
    assert utils.is_constant([1, 1, 1])
    assert utils.is_constant([-1, -1, -1])
    assert not utils.is_constant([1, 1, 1, 0])
    assert not utils.is_constant([1, 1, 1, 1], positive=False)
    assert utils.is_constant([1, 1, 1, 1], positive=True)
    assert utils.is_constant([-1, -1, -1], positive=False)
    assert utils.is_constant([-1, -1, -1], positive=False)


def test__symmetrize():
    assert_allclose(utils._symmetrize([0, 1, 2], [1, 1, 1]),
                    ([-2, -1, 0, 1, 2], [0, 0, 1, 1, 1]))
    assert_allclose(utils._symmetrize([-1, 0, 1, 2], [1, 1, 1, 1]),
                    ([-2, -1, 0, 1, 2], [0, 1, 1, 1, 1]))
    assert_allclose(utils._symmetrize([0.5, 1.5], [1, 1]),
                    ([-1.5, -0.5, 0.5, 1.5], [0, 0, 1, 1]))
    assert_allclose(utils._symmetrize([-0.75, 0.25, 1.25], [1, 1, 1]),
                    ([-0.75, 0.25, 1.25], [1, 1, 1]))
    assert_allclose(utils._symmetrize([-2, -1], [1, 1]),
                    ([-2, -1, 0, 1, 2], [1, 1, 0, 0, 0]))
    assert_allclose(utils._symmetrize([-1.25, -0.25], [1, 1]),
                    ([-1.25, -0.25, 0.75], [1, 1, 0]))
    assert_allclose(utils._symmetrize([0, 0.5, 1], [1, 1, 1]),
                    ([-1, -0.5, 0, 0.5, 1], [0, 0, 1, 1, 1]))
