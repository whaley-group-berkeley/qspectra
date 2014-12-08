from numpy.testing import assert_allclose, assert_equal
import unittest
import numpy as np

from qspectra.hamiltonian import ElectronicHamiltonian
from qspectra.operator_tools import unit_vec
import qspectra.dynamics.liouville_space as liouville_space


class TestSubspaces(unittest.TestCase):
    def test_liouville_subspace_index(self):
        assert_equal(
            liouville_space.liouville_subspace_index('eg,ge', 'ge', 2),
            [1, 2, 3, 6])
        assert_equal(
            liouville_space.liouville_subspace_index('eg,fe', 'gef', 2),
            [1, 2, 7, 11])
        assert_equal(
            liouville_space.liouville_subspace_index('gg,ee,ff', 'gef', 2),
            [0, 5, 6, 9, 10, 15])
        with self.assertRaises(liouville_space.SubspaceError):
            liouville_space.liouville_subspace_index('ef', 'ge', 2)
        # test >1 vibrations:
        assert_equal(
            liouville_space.liouville_subspace_index('gg', 'ge', 1, 2),
            [0, 1, 4, 5])
        assert_equal(
            liouville_space.liouville_subspace_index('eg', 'ge', 1, 2),
            [2, 3, 6, 7])

    def test_all_liouville_subspaces(self):
        self.assertEquals(liouville_space.all_liouville_subspaces('g'),
                          'gg')
        self.assertEquals(liouville_space.all_liouville_subspaces('ge'),
                          'gg,ge,eg,ee')
        self.assertEquals(liouville_space.all_liouville_subspaces('gef'),
                          'gg,ge,gf,eg,ee,ef,fg,fe,ff')


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

    def test_ket_vec(self):
        assert_allclose(self.rho_v,
                        liouville_space.matrix_to_ket_vec(self.rho_d))
        assert_allclose(liouville_space.ket_vec_to_matrix(self.rho_v),
                        self.rho_d)
        assert_allclose(liouville_space.matrix_to_ket_vec(
                            liouville_space.ket_vec_to_matrix(self.rho_v)),
                        self.rho_v)

    def test_bra_vec(self):
        rho_bra_vec = liouville_space.matrix_to_bra_vec(self.rho_d)
        assert_allclose(rho_bra_vec, 0.25 * np.array([1, 2 - 2j, 2 + 2j, 3]))
        operator_vec = liouville_space.matrix_to_ket_vec(self.operator)
        assert_allclose(np.einsum('ij,ji', self.rho_d, self.operator),
                        rho_bra_vec.dot(operator_vec))

    def test_super_matrices(self):
        assert_allclose(
            liouville_space.matrix_to_ket_vec(self.operator.dot(self.rho_d)),
            liouville_space.super_left_matrix(self.operator).dot(self.rho_v))
        assert_allclose(
            liouville_space.matrix_to_ket_vec(self.rho_d.dot(self.operator)),
            liouville_space.super_right_matrix(self.operator).dot(self.rho_v))
        assert_allclose(
            liouville_space.matrix_to_ket_vec(self.operator.dot(self.rho_d)
                                              - self.rho_d.dot(self.operator)),
            liouville_space.super_commutator_matrix(self.operator).dot(self.rho_v))


class ExampleLiouvilleSpaceModel(liouville_space.LiouvilleSpaceModel):
    """
    Example subclass of LiouvilleSpaceModel, since LiouvilleSpaceModel still
    has some abstract methods
    """
    @property
    def evolution_super_operator(self):
        pass

    @property
    def time_step(self):
        pass


class TestLiouvilleSpaceModel(unittest.TestCase):
    def setUp(self):
        self.model = ExampleLiouvilleSpaceModel(ElectronicHamiltonian(np.eye(4)))

    def test_repr(self):
        # should not raise
        repr(self.model)

    def test_thermal_state(self):
        assert_allclose(self.model.thermal_state('gg'), [1])
        assert_allclose(self.model.thermal_state('gg,eg,ge,ee'), unit_vec(0, 25))
        assert_allclose(self.model.thermal_state('ee'), 0.25 * np.eye(4).reshape(-1))

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
