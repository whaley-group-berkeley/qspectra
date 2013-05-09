from numpy.testing import assert_allclose
import numpy as np
import unittest

from spectra import operator_tools
from spectra.utils import MetaArray


class TestSuperOps(unittest.TestCase):
    def setUp(self):
        self.rho_v = 0.25 * np.array([1., 2 + 2j, 2 - 2j, 3])
        self.rho_v_series = MetaArray([np.array([1., 0, 0, 0]), self.rho_v])
        self.rho_d = 0.25 * np.array([[1., 2 - 2j], [2 + 2j, 3]])
        self.M = np.random.rand(2, 2) + 1j * np.random.rand(2, 2)

    def test_vec_den(self):
        assert_allclose(self.rho_v, operator_tools.den_to_vec(self.rho_d))
        assert_allclose(operator_tools.vec_to_den(self.rho_v), self.rho_d)

    def test_vec_pops(self):
        assert_allclose(operator_tools.vec_pops(self.rho_v), [0.25, 0.75])

    def test_vec_exclude_ground(self):
        assert_allclose(operator_tools.vec_exclude_ground(self.rho_v), [0.75])

    def test_time_series_op(self):
        x2 = lambda x: 2 * x
        assert_allclose(operator_tools.time_series_op(x2)(self.rho_v_series),
                        2 * self.rho_v_series)

    def test_series_normalized_pops(self):
        assert_allclose(operator_tools.normalized_pops(self.rho_v_series),
                        [[0], [1]])

    def test_S_left(self):
        assert_allclose(operator_tools.den_to_vec(self.M.dot(self.rho_d)),
                        operator_tools.S_left(self.M).dot(self.rho_v))

    def test_S_right(self):
        assert_allclose(operator_tools.den_to_vec(self.rho_d.dot(self.M)),
                        operator_tools.S_right(self.M).dot(self.rho_v))

    def test_S_commutator(self):
        assert_allclose((operator_tools.S_left(self.M)
                         - operator_tools.S_right(self.M)),
                        operator_tools.S_commutator(self.M))


class TestExtendedStates(unittest.TestCase):
    def setUp(self):
        self.M = np.array([[1., 2 - 2j], [2 + 2j, 3]])

    def test_tensor_to_super(self):
        R_tensor = np.random.rand(2, 2, 2, 2)
        R_super = operator_tools.tensor_to_super(R_tensor)
        for i in xrange(2):
            for j in xrange(2):
                for k in xrange(2):
                    for l in xrange(2):
                        self.assertEquals(R_tensor[i, j, k, l],
                                          R_super[i + 2 * j, k + 2 * l])

    def test_density_subset(self):
        assert_allclose(operator_tools.density_subset('gg,ee', 1), [0, 3])
        assert_allclose(operator_tools.density_subset('gg,ff', 2), [0, 15])
        assert_allclose(operator_tools.density_subset('ge,ef', 2),
                        [4, 8, 13, 14])

    def test_density_pop_indices(self):
        assert_allclose(operator_tools.density_pop_indices(3), [0, 4, 8])

    def test_all_states(self):
        self.assertEquals(operator_tools.all_states(1), [[], [0]])
        self.assertEquals(operator_tools.all_states(2), [[], [0], [1], [0, 1]])
        self.assertEquals(operator_tools.all_states(2, 'ge'), [[], [0], [1]])

    def test_operator_1_to_2(self):
        assert_allclose(operator_tools.operator_1_to_2(self.M), [[4]])
        assert_allclose(operator_tools.operator_1_to_2(self.M, [10]), [[14]])
        assert_allclose(operator_tools.operator_1_to_2(np.diag([1, 10, 100])),
                        np.diag([11, 101, 110]))

    def test_operator_extend(self):
        assert_allclose(operator_tools.operator_extend(self.M, 'e'), self.M)
        assert_allclose(operator_tools.operator_extend(self.M, 'g'), [[0]])
        assert_allclose(operator_tools.operator_extend(self.M, 'f'), [[4]])
        assert_allclose(operator_tools.operator_extend(self.M, 'gg'),
                        [[0, 0], [0, 0]])
        assert_allclose(operator_tools.operator_extend(self.M),
                        [[0, 0, 0, 0],
                         [0, 1, 2 - 2j, 0],
                         [0, 2 + 2j, 3, 0],
                         [0, 0, 0, 4]])

    def test_transition_operator(self):
        assert_allclose(operator_tools.transition_operator(0, 2, 'ge'),
                        [[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        assert_allclose(operator_tools.transition_operator(0, 2),
                        [[0, 1, 0, 0], [1, 0, 0, 0],
                         [0, 0, 0, 1], [0, 0, 1, 0]])
        assert_allclose(operator_tools.transition_operator(2, 2),
                        np.zeros((4, 4)))
