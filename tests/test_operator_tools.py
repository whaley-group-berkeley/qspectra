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


class TestExtendedStates(unittest.TestCase):
    def setUp(self):
        self.M = np.array([[1., 2 - 2j], [2 + 2j, 3]])

    def test_all_states(self):
        self.assertEquals(operator_tools.all_states(1), [[], [0]])
        self.assertEquals(operator_tools.all_states(2), [[], [0], [1], [0, 1]])
        self.assertEquals(operator_tools.all_states(2, 'ge'), [[], [0], [1]])

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
        self.assertItemsEqual(operator_tools.extract_subspace('gg,eg->gg'),
                              'ge')
        self.assertItemsEqual(operator_tools.extract_subspace('gg,ee,ff'),
                              'gef')

    def test_hilbert_space_index(self):
        self.assertEquals(operator_tools.hilbert_subspace_index('g', 'gef', 3),
                          slice(0, 1))
        self.assertEquals(operator_tools.hilbert_subspace_index('e', 'gef', 3),
                          slice(1, 4))
        self.assertEquals(operator_tools.hilbert_subspace_index('f', 'gef', 3),
                          slice(4, 7))
        self.assertEquals(operator_tools.hilbert_subspace_index('f', 'ef', 3),
                          slice(3, 6))
