from numpy.testing import assert_allclose, assert_equal
import numpy as np
import unittest

from qspectra import operator_tools


class TestVibrationOperators(unittest.TestCase):
    def test(self):
        assert_allclose(operator_tools.vib_annihilate(3),
                        [[0, 1, 0], [0, 0, np.sqrt(2)], [0, 0, 0]])
        assert_allclose(operator_tools.vib_create(3),
                        [[0, 0, 0], [1, 0, 0], [0, np.sqrt(2), 0]])
        assert_allclose(operator_tools.vib_create(3).dot(
                            operator_tools.vib_annihilate(3)),
                        [[0, 0, 0], [0, 1, 0], [0, 0, 2]])


def test_unit_vec():
    assert_allclose(operator_tools.unit_vec(0, 3), [1, 0, 0])


class TestBasisTransform(unittest.TestCase):
    def setUp(self):
        self.U = np.array([[1, 1], [-1, 1]]) / np.sqrt(2)

    def test_basis_transform_operator(self):
        X = np.random.randn(4, 4)
        for U in [np.eye(4), np.eye(2)]:
            X_prime = operator_tools.basis_transform_operator(X, U)
            assert_allclose(X, X_prime)
        with self.assertRaises(ValueError):
            operator_tools.basis_transform_operator(X, np.eye(3))
        with self.assertRaises(ValueError):
            operator_tools.basis_transform_operator(X, np.ones((1, 2)))

        X = np.eye(2)
        actual = operator_tools.basis_transform_operator(X, self.U)
        assert_allclose(actual, X, atol=1e-15)

        X = np.array([[1, -1], [-1, 1]]) / 2.0
        actual = operator_tools.basis_transform_operator(X, self.U)
        expected = np.array([[1, 0], [0, 0]])
        assert_allclose(actual, expected, atol=1e-15)

    def test_basis_transform_vector(self):
        for rho in [np.random.randn(4),
                    np.random.randn(3, 4),
                    np.random.randn(1, 2, 3, 4)]:
            for U in [np.eye(4), np.eye(2)]:
                rho_prime = operator_tools.basis_transform_vector(rho, U)
                assert_allclose(rho, rho_prime)

        rho = np.random.randn(3)
        U = np.ones((3, 2))
        with self.assertRaises(ValueError):
            operator_tools.basis_transform_vector(rho, U)

        rho = [1, 0]
        actual = operator_tools.basis_transform_vector(rho, self.U)
        expected = np.sqrt([0.5, 0.5])
        assert_allclose(actual, expected)

        rho = [0.5, -0.5, -0.5, 0.5]
        actual = operator_tools.basis_transform_vector(rho, self.U)
        expected = np.array([1, 0, 0, 0])
        assert_allclose(actual, expected)


class TestExtendedStates(unittest.TestCase):
    def setUp(self):
        self.M = np.array([[1., 2 - 2j], [2 + 2j, 3]])

    def test_all_states(self):
        self.assertEqual(operator_tools.all_states(1), [[], [0]])
        self.assertEqual(operator_tools.all_states(2), [[], [0], [1], [0, 1]])
        self.assertEqual(operator_tools.all_states(2, 'ge'), [[], [0], [1]])

    def test_operator_1_to_2(self):
        assert_allclose(operator_tools.operator_1_to_2(self.M), [[4]])
        assert_allclose(operator_tools.operator_1_to_2(np.diag([1, 10, 100])),
                        np.diag([11, 101, 110]))

    def test_operator_extend(self):
        assert_allclose(operator_tools.operator_extend(self.M, 'e'), self.M)
        assert_allclose(operator_tools.operator_extend(self.M, 'g'), [[0]])
        assert_allclose(operator_tools.operator_extend(self.M, 'f'), [[4]])
        assert_allclose(operator_tools.operator_extend(self.M),
                        [[0, 0, 0, 0],
                         [0, 1, 2 - 2j, 0],
                         [0, 2 + 2j, 3, 0],
                         [0, 0, 0, 4]])

    def test_transition_operator(self):
        assert_allclose(operator_tools.transition_operator(0, 2, 'ge'),
                        [[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        assert_allclose(operator_tools.transition_operator(0, 2),
                        [[0, 1, 0, 0],
                         [1, 0, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 1, 0]])
        assert_allclose(operator_tools.transition_operator(0, 2, 'gef', ''),
                        np.zeros((4, 4)))
        minus = operator_tools.transition_operator(0, 2, 'gef', '-')
        assert_allclose(minus,
                        [[0, 1, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 0, 0]])
        assert_allclose(minus.dot([0, 0, 0, 0]), [0, 0, 0, 0])
        assert_allclose(minus.dot([0, 1, 0, 0]), [1, 0, 0, 0])
        assert_allclose(minus.dot([0, 0, 0, 1]), [0, 0, 1, 0])
        assert_allclose(minus.conj().T.dot([1, 0, 0, 0]), [0, 1, 0, 0])
        assert_allclose(operator_tools.transition_operator(0, 2, 'gef', '+'),
                        [[0, 0, 0, 0],
                         [1, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 1, 0]])


class TestSubspaces(unittest.TestCase):
    def test_n_excitations(self):
        assert_equal(operator_tools.n_excitations(1), [1, 1, 0])
        assert_equal(operator_tools.n_excitations(2), [1, 2, 1])
        assert_equal(operator_tools.n_excitations(3), [1, 3, 3])
        assert_equal(operator_tools.n_excitations(3, 2), [2, 6, 6])

    def test_extract_subspace(self):
        self.assertEqual(operator_tools.extract_subspace('gg,eg->gg'),
                         ['g', 'e'])
        self.assertEqual(operator_tools.extract_subspace('gg,ee,ff'),
                         ['g', 'e', 'f'])

    def test_full_liouville_subspace(self):
        self.assertEqual(operator_tools.full_liouville_subspace('gg'), 'gg')
        self.assertEqual(operator_tools.full_liouville_subspace('ee'), 'ee')
        self.assertEqual(operator_tools.full_liouville_subspace('ge,ee'),
                         'gg,ge,eg,ee')
        self.assertEqual(operator_tools.full_liouville_subspace('ge,ee,ff'),
                         'gg,ge,gf,eg,ee,ef,fg,fe,ff')

    def test_hilbert_space_index(self):
        self.assertEqual(operator_tools.hilbert_subspace_index('g', 'gef', 3),
                          slice(0, 1))
        self.assertEqual(operator_tools.hilbert_subspace_index('e', 'gef', 3),
                          slice(1, 4))
        self.assertEqual(operator_tools.hilbert_subspace_index('f', 'gef', 3),
                          slice(4, 7))
        self.assertEqual(operator_tools.hilbert_subspace_index('f', 'ef', 3),
                          slice(3, 6))
