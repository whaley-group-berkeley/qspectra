import numpy as np
import unittest
from numpy.testing import assert_allclose

from qspectra import hamiltonian, GAUSSIAN_SD_FWHM


class SharedTests(object):
    def test_rotating_frame(self):
        H_rw = self.H_sys.in_rotating_frame(2)
        self.assertEqual(H_rw.energy_offset, 2)
        self.assertEqual(self.H_sys.mean_excitation_freq,
                         H_rw.mean_excitation_freq)
        # verify idempotent:
        H_rw2 = H_rw.in_rotating_frame(3)
        self.assertEqual(H_rw2.energy_offset, 3)
        self.assertEqual(self.H_sys.mean_excitation_freq,
                         H_rw2.mean_excitation_freq)

    def test_sample_ensemble(self):
        H_sampled, = self.H_sys.sample_ensemble(1)
        self.assertEqual(H_sampled.ref_system, self.H_sys)
        # verify time-steps and excitation frequencies remain fixed
        self.assertEqual(H_sampled.mean_excitation_freq,
                         self.H_sys.mean_excitation_freq)
        self.assertEqual(H_sampled.freq_step, self.H_sys.freq_step)
        self.assertEqual(H_sampled.time_step, self.H_sys.time_step)
        # verify sample_ensemble and in_rotating_frame commute:
        H_rw = self.H_sys.in_rotating_frame(10)
        H_sampled_rw = H_sampled.in_rotating_frame(10)
        H_rw_sampled, = H_rw.sample_ensemble(1)
        assert_allclose(H_sampled_rw.H('gef'), H_rw_sampled.H('gef'))
        self.assertEqual(H_sampled_rw.time_step, H_rw.time_step)
        self.assertEqual(H_rw_sampled.time_step, H_rw.time_step)
        self.assertEqual(self.H_sys.mean_excitation_freq,
                         H_sampled_rw.mean_excitation_freq)
        self.assertEqual(self.H_sys.mean_excitation_freq,
                         H_rw_sampled.mean_excitation_freq)


class TestElectronicHamiltonian(unittest.TestCase, SharedTests):
    def setUp(self):
        self.M = np.array([[1., 0], [0, 3]])
        self.H_sys = hamiltonian.ElectronicHamiltonian(
            self.M, bath=None, dipoles=[[1, 0, 0], [0, 1, 0]], disorder=1,
            energy_spread_extra=1.0)

    def test_properties(self):
        self.assertEqual(self.H_sys.energy_spread_extra, 1)
        self.assertEqual(self.H_sys.n_sites, 2)
        self.assertEqual(self.H_sys.n_states('gef'), 4)
        self.assertEqual(self.H_sys.freq_step, 10.0)
        self.assertEqual(self.H_sys.time_step, 0.1)
        assert_allclose(self.H_sys.H('e'), self.M)
        assert_allclose(self.H_sys.E('g'), [0])
        assert_allclose(self.H_sys.E('ge'), [0, 1, 3])
        assert_allclose(self.H_sys.E('gef'), [0, 1, 3, 4])
        assert_allclose(self.H_sys.ground_state('ge'),
                        [[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.assertEqual(self.H_sys.mean_excitation_freq, 2)
        assert_allclose(self.H_sys.number_operator(1, 'gef'),
                        np.diag([0, 0, 1, 1]))
        assert_allclose(self.H_sys.number_operator(0, 'ge'),
                        np.diag([0, 1, 0]))
        assert_allclose(self.H_sys.dipole_operator('gef', 'x', '-+'),
                        [[0, 1, 0, 0], [1, 0, 0, 0],
                         [0, 0, 0, 1], [0, 0, 1, 0]])
        H_no_dipoles = hamiltonian.ElectronicHamiltonian(self.M)
        with self.assertRaises(hamiltonian.HamiltonianError):
            H_no_dipoles.dipole_operator()
        with self.assertRaises(hamiltonian.HamiltonianError):
            self.H_sys.system_bath_couplings()

    def test_rw_properties(self):
        H_rw = self.H_sys.in_rotating_frame()
        assert_allclose(H_rw.H('e'), [[-1, 0], [0, 1]])
        self.assertItemsEqual(H_rw.E('gef'), [0, 1, -1, 0])
        self.assertEqual(H_rw.energy_offset, 2)
        self.assertEqual(H_rw.mean_excitation_freq, 2)
        self.assertEqual(H_rw.freq_step, 6.0)

    def test_more_sample_ensemble(self):
        H_sampled_ori, = self.H_sys.sample_ensemble(1, random_orientations=True)
        self.assertAlmostEqual(np.dot(*H_sampled_ori.dipoles), 0)

        H_sampled, = self.H_sys.sample_ensemble(1)
        disorder = lambda random_state: np.diag(GAUSSIAN_SD_FWHM * random_state.randn(2))
        H_sampled_matching_seed, = hamiltonian.ElectronicHamiltonian(
            self.M, disorder=disorder, random_seed=0).sample_ensemble(1)
        H_sampled_non_matching_seed, = hamiltonian.ElectronicHamiltonian(
            self.M, disorder=disorder, random_seed=[1, 2, 3]).sample_ensemble(1)
        assert_allclose(H_sampled.H('gef'), H_sampled_matching_seed.H('gef'))
        self.assertFalse(np.allclose(H_sampled.H('gef'),
                                     H_sampled_non_matching_seed.H('gef')))

    def test_thermal_state(self):
        assert_allclose(hamiltonian.thermal_state(self.H_sys.H_1exc, 2),
                        1 / (np.exp(0.5) + np.exp(-0.5)) *
                        np.array([[np.exp(0.5), 0], [0, np.exp(-0.5)]]))


class DummyBath(object):
    temperature = 2


class TestVibronicHamiltonian(unittest.TestCase, SharedTests):
    def setUp(self):
        H_E = hamiltonian.ElectronicHamiltonian([[1.0]], bath=DummyBath(),
                                                dipoles=[[1, 0, 0], [0, 1, 0]])
        self.H_sys = hamiltonian.VibronicHamiltonian(H_E, [2], [10], [[5]])

    def test_properties(self):
        self.assertEqual(self.H_sys.n_sites, 1)
        self.assertEqual(self.H_sys.n_vibrational_states, 2)
        self.assertEqual(self.H_sys.n_states('gef'), 4)
        assert_allclose(self.H_sys.H('ge'),
                        [[0, 0, 0, 0],
                         [0, 10, 0, 0],
                         [0, 0, 1, 5],
                         [0, 0, 5, 11]])
        self.assertAlmostEqual(self.H_sys.mean_excitation_freq, 6)
        assert_allclose(self.H_sys.ground_state('g'),
                        1 / (1 + np.exp(-5)) * np.diag([1, np.exp(-5)]))

    def test_rw_properties(self):
        with self.assertRaises(hamiltonian.HamiltonianError):
            self.H_sys.energy_offset = 10

    def test_operators(self):
        assert_allclose(self.H_sys.system_bath_couplings('ge'),
                        [[[0, 0, 0, 0], [0, 0, 0, 0],
                          [0, 0, 1, 0], [0, 0, 0, 1]]])
