from numpy.testing import assert_allclose
from mockito import mock, when
import numpy as np
import unittest

from spectra import hamiltonian
from spectra.utils import MetaArray


class TestHamiltonian(unittest.TestCase):
    def setUp(self):
        self.M = np.random.rand(2, 2) + 1j * np.random.rand(2, 2)

    def test_vec_den(self):
        assert_allclose(self.rho_v, operator_tools.den_to_vec(self.rho_d))
        assert_allclose(operator_tools.vec_to_den(self.rho_v), self.rho_d)


class TestElectronicHamiltonian(unittest.TestCase):
    def setUp(self):
        self.M = np.array([[1., 0], [0, 3]])

    def test(self):
        H_el = hamiltonian.ElectronicHamiltonian(self.M)
        self.assertEqual(H_el.n_sites, 2)
        self.assertEqual(H_el.n_states('gef'), 4)
        assert_allclose(H_el.H('e'), self.M)
        assert_allclose(H_el.in_rotating_frame(2).H('e'),
                        [[-1, 0], [0, 1]])

        assert_allclose(H_el.thermal_state(1 / np.log(2)), )


        assert_allclose(self.rho_v, operator_tools.den_to_vec(self.rho_d))
        assert_allclose(operator_tools.vec_to_den(self.rho_v), self.rho_d)
