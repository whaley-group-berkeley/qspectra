from numpy.testing import assert_allclose, assert_equal
import unittest
import numpy as np

import qspectra.dynamical_models.liouville_space as liouville_space


class TestSubspaces(unittest.TestCase):
    def test_n_excitations(self):
        self.assertEquals(liouville_space.n_excitations(1), [1, 1, 0])
        self.assertEquals(liouville_space.n_excitations(2), [1, 2, 1])
        self.assertEquals(liouville_space.n_excitations(3), [1, 3, 3])

    def test_n_states_from_n_sites(self):
        self.assertEquals(liouville_space.n_states_from_n_sites(1), 2)
        self.assertEquals(liouville_space.n_states_from_n_sites(2), 4)
        self.assertEquals(liouville_space.n_states_from_n_sites(3), 7)
        self.assertEquals(liouville_space.n_states_from_n_sites(3, 'g'), 1)
        self.assertEquals(liouville_space.n_states_from_n_sites(3, 'ef'), 6)
        self.assertEquals(liouville_space.n_states_from_n_sites(3, 'ge'), 4)

    def test_n_sites_from_n_states(self):
        self.assertEquals(liouville_space.n_sites_from_n_states(2), 1)
        self.assertEquals(liouville_space.n_sites_from_n_states(4), 2)
        self.assertEquals(liouville_space.n_sites_from_n_states(7), 3)
        self.assertEquals(liouville_space.n_sites_from_n_states(6, 'ef'), 3)
        self.assertEquals(liouville_space.n_sites_from_n_states(4, 'ge'), 3)
        with self.assertRaises(liouville_space.SubspaceError):
            liouville_space.n_sites_from_n_states(1, 'g')

    def test_extract_subspace(self):
        self.assertItemsEqual(liouville_space.extract_subspace('gg,eg->gg'),
                              'ge')
        self.assertItemsEqual(liouville_space.extract_subspace('gg,ee,ff'),
                              'gef')

    def test_liouville_subspace_indices(self):
        assert_equal(
            liouville_space.liouville_subspace_indices('eg,ge', 'ge', 2),
            [1, 2, 3, 6])
        assert_equal(
            liouville_space.liouville_subspace_indices('eg,fe', 'gef', 2),
            [1, 2, 7, 11])
        assert_equal(
            liouville_space.liouville_subspace_indices('gg,ee,ff', 'gef', 2),
            [0, 5, 6, 9, 10, 15])
        with self.assertRaises(liouville_space.SubspaceError):
            liouville_space.liouville_subspace_indices('ef', 'ge', 2)


class TestSuperOperators(unittest.TestCase):
    def setUp(self):
        self.rho_v = 0.25 * np.array([1., 2 + 2j, 2 - 2j, 3])
        self.rho_d = 0.25 * np.array([[1., 2 - 2j], [2 + 2j, 3]])
        self.operator = np.random.rand(2, 2) + 1j * np.random.rand(2, 2)

    def test_tensor_to_super(self):
        R_tensor = np.random.rand(2, 2, 2, 2)
        R_super = liouville_space.tensor_to_super(R_tensor)
        for i in xrange(2):
            for j in xrange(2):
                for k in xrange(2):
                    for l in xrange(2):
                        self.assertEquals(R_tensor[i, j, k, l],
                                          R_super[i + 2 * j, k + 2 * l])

    def test_vec_den(self):
        assert_allclose(self.rho_v, liouville_space.den_to_vec(self.rho_d))
        assert_allclose(liouville_space.vec_to_den(self.rho_v), self.rho_d)
        assert_allclose(liouville_space.den_to_vec(
            liouville_space.vec_to_den(self.rho_v)), self.rho_v)

    def test_super_matrices(self):
        assert_allclose(
            liouville_space.den_to_vec(self.operator.dot(self.rho_d)),
            liouville_space.super_left_matrix(self.operator).dot(self.rho_v))
        assert_allclose(
            liouville_space.den_to_vec(self.rho_d.dot(self.operator)),
            liouville_space.super_right_matrix(self.operator).dot(self.rho_v))
        assert_allclose(
            liouville_space.den_to_vec(self.operator.dot(self.rho_d)
                                       - self.rho_d.dot(self.operator)),
            liouville_space.super_commutator_matrix(self.operator).dot(self.rho_v))


class TestLiouvilleSpaceOperator(unittest.TestCase):
    def test(self):
        X = np.array([[1, 2], [3, 4]])
        L = liouville_space.LiouvilleSpaceOperator(X, 'ge', 'gg,eg,ge,ee->gg')
        assert_allclose(L.left_multiply([1, 10, 100, 1000]), [21])
        assert_allclose(L.right_multiply([1, 10, 100, 1000]), [301])
        assert_allclose(L.commutator([1, 10, 100, 1000]), [-280])
        assert_allclose(L.expectation_value([1, 10, 100, 1000]), [21])
        L = liouville_space.LiouvilleSpaceOperator(X, 'ge', 'ee->gg,ee')
        assert_allclose(L.left_multiply([1]), [0, 4])
        assert_allclose(L.expectation_value([1]), 4)
        L = liouville_space.LiouvilleSpaceOperator(X, 'ge', 'ee->gg,ge,eg,ee')
        assert_allclose(L.expectation_value([1]), 4)
