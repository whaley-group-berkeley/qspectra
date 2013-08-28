from numpy.testing import assert_allclose, assert_equal
import unittest
import numpy as np

from qspectra.hamiltonian import ElectronicHamiltonian
from qspectra.operator_tools import unit_vec
import qspectra.dynamical_models.liouville_space as liouville_space


class TestSubspaces(unittest.TestCase):
    def test_n_excitations(self):
        assert_equal(liouville_space.n_excitations(1), [1, 1, 0])
        assert_equal(liouville_space.n_excitations(2), [1, 2, 1])
        assert_equal(liouville_space.n_excitations(3), [1, 3, 3])

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
        # test >1 vibrations:
        assert_equal(
            liouville_space.liouville_subspace_indices('gg', 'ge', 1, 2),
            [0, 1, 4, 5])
        assert_equal(
            liouville_space.liouville_subspace_indices('eg', 'ge', 1, 2),
            [2, 3, 6, 7])


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


class ExampleLiouvilleSpaceModel(liouville_space.LiouvilleSpaceModel):
    """
    Example subclass of LiouvilleSpaceModel, since LiouvilleSpaceModel still
    has some abstract methods
    """
    def equation_of_motion(self, liouville_subspace):
        pass

    @property
    def time_step(self):
        pass


class TestLiouvilleSpaceModel(unittest.TestCase):
    def setUp(self):
        self.model = ExampleLiouvilleSpaceModel(ElectronicHamiltonian(np.eye(4)))

    def test_ground_state(self):
        assert_allclose(self.model.ground_state('gg'), [1])
        assert_allclose(self.model.ground_state('gg,eg,ge,ee'), unit_vec(0, 25))
        assert_allclose(self.model.ground_state('ee'), np.zeros(16))

    def test_map_between_subspaces(self):
        assert_allclose(
            self.model.map_between_subspaces(np.ones(25), 'gg,eg,ge,ee', 'gg'),
            [1])
        assert_allclose(
            self.model.map_between_subspaces(np.ones(25), 'gg,eg,ge,ee', 'gf'),
            np.zeros(6))
        assert_allclose(
            self.model.map_between_subspaces(np.ones(25), 'gg,eg,ge,ee', 'gg,gf'),
            np.concatenate([[1], np.zeros(6)]))


class TestLiouvilleSpaceOperator(unittest.TestCase):
    def test(self):
        X = np.array([[1, 2], [3, 4]])
        model = ExampleLiouvilleSpaceModel(ElectronicHamiltonian([[0]]),
                                           hilbert_subspace='ge')
        L = liouville_space.LiouvilleSpaceOperator(X, 'gg,eg,ge,ee->gg', model)
        assert_allclose(L.left_multiply([1, 10, 100, 1000]), [21])
        assert_allclose(L.right_multiply([1, 10, 100, 1000]), [301])
        assert_allclose(L.commutator([1, 10, 100, 1000]), [-280])
        assert_allclose(L.expectation_value([1, 10, 100, 1000]), [21])
        L = liouville_space.LiouvilleSpaceOperator(X, 'ee->gg,ee', model)
        assert_allclose(L.left_multiply([1]), [0, 4])
        assert_allclose(L.expectation_value([1]), 4)
        L = liouville_space.LiouvilleSpaceOperator(X, 'ee->gg,ge,eg,ee', model)
        assert_allclose(L.expectation_value([1]), 4)
