import unittest
import warnings

import numpy as np
from numpy.testing import assert_allclose

from qspectra import hamiltonian, GAUSSIAN_SD_FWHM


class TestCheckHermitian(unittest.TestCase):
    def test(self):
        hamiltonian.check_hermitian([[1, 0], [0, -1]])
        hamiltonian.check_hermitian([[1, 1j], [-1j, -1]])
        with self.assertRaises(ValueError):
            hamiltonian.check_hermitian([[1, 1j], [1j, -1]])
            hamiltonian.check_hermitian([[1, 1], [-1, -1]])


class SharedTests(object):
    def test_rotating_frame(self):
        self.assertEqual(self.H_sys, self.H_sys.in_rotating_frame(0))
        H_rw = self.H_sys.in_rotating_frame(2)
        self.assertEqual(H_rw.rw_freq, 2)
        self.assertNotEqual(H_rw, self.H_sys)
        assert_allclose(H_rw.E('e') + 2, self.H_sys.E('e'))
        # test default:
        H_rw_default = self.H_sys.in_rotating_frame()
        self.assertEqual(H_rw_default.rw_freq, self.H_sys.transition_energy)
        assert_allclose(H_rw_default.E('e'),
                        self.H_sys.E('e') - self.H_sys.transition_energy)
        # verify idempotent:
        H_rw2 = H_rw.in_rotating_frame(3)
        self.assertEqual(H_rw2.rw_freq, 3)
        assert_allclose(H_rw2.E('e') + 3, self.H_sys.E('e'))
        self.assertNotEqual(H_rw, H_rw2)
        self.assertEqual(H_rw, H_rw.in_rotating_frame(2))
        self.assertEqual(H_rw, H_rw.in_rotating_frame(2).in_rotating_frame(2))

    def test_sample(self):
        H_sampled = self.H_sys.sample(0)
        self.assertNotEqual(H_sampled, self.H_sys)
        self.assertNotEqual(H_sampled, self.H_sys.sample())
        # verify time-steps and excitation frequencies remain fixed
        self.assertEqual(H_sampled.freq_step, self.H_sys.freq_step)
        self.assertEqual(H_sampled.time_step, self.H_sys.time_step)
        # verify sample_ensemble and in_rotating_frame commute:
        H_rw = self.H_sys.in_rotating_frame(10)
        H_sampled_rw = H_sampled.in_rotating_frame(10)
        H_rw_sampled = H_rw.sample(0)
        self.assertEqual(H_sampled_rw, H_rw_sampled)
        self.assertNotEqual(H_rw_sampled, H_rw.sample(1))
        assert_allclose(H_sampled_rw.H('gef'), H_rw_sampled.H('gef'))
        self.assertEqual(H_sampled_rw.time_step, H_rw.time_step)
        self.assertEqual(H_rw_sampled.time_step, H_rw.time_step)

    def test_sample_ensemble(self):
        self.assertEqual(list(self.H_sys.sample_ensemble(3)),
                         [self.H_sys.sample(n) for n in range(3)])

    def test_state_consistency(self):
        for state in (self.H_sys.ground_state('gef'),
                      self.H_sys.thermal_state('gef')):
            hamiltonian.check_hermitian(state)
            self.assertAlmostEqual(np.trace(state), 1)


class DummyBath(object):
    temperature = 2


def construct_elec(M):
    return hamiltonian.ElectronicHamiltonian(
        M, bath=None, dipoles=[[1, 0, 0], [0, 1, 0]],
        disorder=1, energy_spread_extra=1.0)


class TestElectronicHamiltonian(unittest.TestCase, SharedTests):
    def setUp(self):
        self.M = np.array([[1., 0], [0, 3]])
        self.H_sys = construct_elec(self.M)

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
        self.assertEqual(self.H_sys.transition_energy, 2)
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

    def test_eq(self):
        self.assertEqual(self.H_sys, construct_elec(self.M))
        self.assertNotEqual(self.H_sys,
            construct_elec(self.M + np.eye(2)).in_rotating_frame(1))
        self.assertNotEqual(self.H_sys.sample(1),
            construct_elec(self.M + np.eye(2)).in_rotating_frame(1).sample(1))
        self.assertNotEqual(self.H_sys.sample(1),
            construct_elec(self.M + np.eye(2)).sample(1).in_rotating_frame(1))
        self.assertNotEqual(self.H_sys, construct_elec(2 * self.M))

    def test_rw_properties(self):
        H_rw = self.H_sys.in_rotating_frame()
        assert_allclose(H_rw.H('e'), [[-1, 0], [0, 1]])
        assert_allclose(H_rw.E('gef'), [0, -1, 1, 0])
        self.assertEqual(H_rw.transition_energy, 0)
        self.assertEqual(H_rw.freq_step, 4.0)
        assert_allclose(self.H_sys.in_rotating_frame(3).H('e'),
                        [[-2, 0], [0, 0]])

    def test_more_sample_ensemble(self):
        H_sampled_ori = self.H_sys.sample(1, random_orientations=True)
        self.assertAlmostEqual(np.dot(*H_sampled_ori.dipoles), 0)

        H_sampled = self.H_sys.sample(1)
        disorder = lambda random_state: np.diag(GAUSSIAN_SD_FWHM * random_state.randn(2))
        H_sampled_matching_seed = hamiltonian.ElectronicHamiltonian(
            self.M, disorder=disorder, random_seed=0).sample(1)
        H_sampled_non_matching_seed = hamiltonian.ElectronicHamiltonian(
            self.M, disorder=disorder, random_seed=[1, 2, 3]).sample(1)
        assert_allclose(H_sampled.H('gef'), H_sampled_matching_seed.H('gef'))
        self.assertNotEqual(H_sampled, H_sampled_matching_seed)
        self.assertNotEqual(H_sampled_non_matching_seed, H_sampled_matching_seed)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            hamiltonian.ElectronicHamiltonian(self.M).sample()
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, RuntimeWarning))

    def test_ground_state(self):
        assert_allclose(self.H_sys.ground_state('g'), [[1]])
        assert_allclose(self.H_sys.ground_state('e'), [[1, 0], [0, 0]])
        assert_allclose(self.H_sys.ground_state('ge'),
                        [[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        H_degenerate = construct_elec([[1, 0], [0, 1]])
        assert_allclose(H_degenerate.ground_state('e'), [[0.5, 0], [0, 0.5]])

    def test_thermal_state(self):
        assert_allclose(self.H_sys.thermal_state('e'), [[1, 0], [0, 0]])
        H_with_bath = construct_elec(self.M)
        H_with_bath.bath = DummyBath()
        assert_allclose(H_with_bath.thermal_state('e'),
                        1 / (np.exp(0.5) + np.exp(-0.5)) *
                        np.array([[np.exp(0.5), 0], [0, np.exp(-0.5)]]))
        H_with_bath.bath.temperature = 1e-25
        with self.assertRaises(OverflowError):
            print H_with_bath.thermal_state('gef')

    def test_basis_labels(self):
        self.assertEqual(self.H_sys.basis_labels('gef', braket=True),
            ['|00>', '|10>', '|01>', '|11>'])
        self.assertEqual(self.H_sys.basis_labels('gef'),
            ['00', '10', '01', '11'])

        H_sys_labeled = hamiltonian.ElectronicHamiltonian(
        self.M, bath=None, dipoles=[[1, 0, 0], [0, 1, 0]],
        disorder=1, energy_spread_extra=1.0, site_labels=["one", "two"])

        self.assertEqual(H_sys_labeled.basis_labels('gef', braket=True),
            ['|g>', '|one>', '|two>', '|one,two>'])
        self.assertEqual(H_sys_labeled.basis_labels('gef'),
            ['g', 'one', 'two', 'one,two'])

class TestVibronicHamiltonian(unittest.TestCase, SharedTests):
    def setUp(self):
        H_E = hamiltonian.ElectronicHamiltonian([[1.0]], bath=DummyBath(),
                                                dipoles=[[1, 0, 0], [0, 1, 0]],
                                                disorder=0)
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
        self.assertAlmostEqual(self.H_sys.transition_energy, 6)
        assert_allclose(self.H_sys.thermal_state('g'),
                        1 / (1 + np.exp(-5)) * np.diag([1, np.exp(-5)]))

    def test_operators(self):
        assert_allclose(self.H_sys.system_bath_couplings('ge'),
                        [[[0, 0, 0, 0], [0, 0, 0, 0],
                          [0, 0, 1, 0], [0, 0, 0, 1]]])

    def test_basis_labels(self):
        self.assertEqual(self.H_sys.basis_labels('gef', braket=1),
            ['|0>|0>', '|0>|1>', '|1>|0>', '|1>|1>'])
        self.assertEqual(self.H_sys.basis_labels('gef'),
            [('0', '0'), ('0', '1'), ('1', '0'), ('1', '1')])

        H_E = hamiltonian.ElectronicHamiltonian([[1.0]], bath=DummyBath(),
                                                dipoles=[[1, 0, 0], [0, 1, 0]],
                                                disorder=0, site_labels=["one"])
        H_sys_labeled = hamiltonian.VibronicHamiltonian(H_E, [2], [10], [[5]])

        self.assertEqual(H_sys_labeled.basis_labels('gef', braket=1),
            ['|g>|0>', '|g>|1>', '|one>|0>', '|one>|1>'])
        self.assertEqual(H_sys_labeled.basis_labels('gef'),
            [('g', '0'), ('g', '1'), ('one', '0'), ('one', '1')])

