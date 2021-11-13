from numpy.testing import assert_allclose
from unittest import TestCase
import numpy as np

from qspectra import pulse
from qspectra.constants import GAUSSIAN_SD_FWHM


class TestPulse(TestCase):
    def test(self):
        self.assertRaisesRegex(TypeError, "Can't instantiate abstract class",
                               pulse.Pulse)


class TestCustomPulse(TestCase):
    def test(self):
        pump = pulse.CustomPulse(0, 1, lambda x, r: x + r)
        self.assertEqual(pump.t_init, 0)
        self.assertEqual(pump.t_final, 1)
        self.assertEqual(pump(1, 2), 3)


class TestGaussianPulse(TestCase):
    def test1(self):
        pump = pulse.GaussianPulse(carrier_freq=10, fwhm=1, t_peak=5, scale=100,
                                   freq_convert=1, t_limits_multiple=2)
        assert_allclose(pump(np.array([5, 5.5]), 10), [100, 50])
        assert_allclose(pump(np.array([5, 5.5]), 5), [100, 50 * np.exp(2.5j)])
        self.assertEqual(pump.t_init, 5 - 2 * GAUSSIAN_SD_FWHM)
        self.assertEqual(pump.t_final, 5 + 2 * GAUSSIAN_SD_FWHM)

    def test2(self):
        pump = pulse.GaussianPulse(carrier_freq=10, fwhm=2, t_peak=0,
                                    scale=100, freq_convert=2)
        assert_allclose(pump(np.array([0, 1]), 10.0), [100, 50])
        assert_allclose(pump(1, 0.0), 50 * np.exp(20.0j))
