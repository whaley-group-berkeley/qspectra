import numpy as np
import unittest
from numpy.testing import assert_allclose

from qspectra import hamiltonian


class TestElectronicHamiltonian(unittest.TestCase):
    def setUp(self):
        self.M = np.array([[1., 0], [0, 3]])
        self.H_el = hamiltonian.ElectronicHamiltonian(self.M, 0, None, None, 1.0)

    def test_properties(self):
        self.assertEqual(self.H_el.energy_offset, 0)
        self.assertEqual(self.H_el.energy_spread_extra, 1)
        self.assertEqual(self.H_el.n_sites, 2)
        self.assertEqual(self.H_el.n_states('gef'), 4)
        self.assertEqual(self.H_el.freq_step, 10.0)
        self.assertEqual(self.H_el.time_step, 0.1)
        assert_allclose(self.H_el.H('e'), self.M)
        assert_allclose(self.H_el.E('g'), [0])
        assert_allclose(self.H_el.E('ge'), [0, 1, 3])
        assert_allclose(self.H_el.E('gef'), [0, 1, 3, 4])
        self.assertEqual(self.H_el.mean_excitation_freq, 2)
        with self.assertRaises(hamiltonian.HamiltonianError):
            self.H_el.dipole_operator()
        with self.assertRaises(hamiltonian.HamiltonianError):
            self.H_el.system_bath_couplings()

    def test_rotating_frame(self):
        H_rw = self.H_el.in_rotating_frame(2)
        assert_allclose(H_rw.H('e'), [[-1, 0], [0, 1]])
        self.assertItemsEqual(H_rw.E('gef'), [0, 1, -1, 0])
        self.assertEqual(H_rw.energy_offset, 2)
        self.assertEqual(H_rw.mean_excitation_freq, 2)
        self.assertEqual(H_rw.freq_step, 6.0)

        H_rw2 = self.H_el.in_rotating_frame(3)
        self.assertEqual(H_rw2.energy_offset, 3)
        self.assertEqual(H_rw2.mean_excitation_freq, 2)
        assert_allclose(H_rw2.H('e'), [[-2, 0], [0, 0]])

    def test_thermal_state(self):
        assert_allclose(hamiltonian.thermal_state(self.H_el.H_1exc, 2),
                        1 / (np.exp(0.5) + np.exp(-0.5)) *
                        np.array([[np.exp(0.5), 0], [0, np.exp(-0.5)]]))
