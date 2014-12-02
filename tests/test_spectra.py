import numpy as np
import unittest
import itertools

from qspectra import (ElectronicHamiltonian, DebyeBath, CM_K, CM_FS,
                      RedfieldModel, absorption_spectra)
from qspectra.simulate.utils import is_constant


class TestRedfieldMonomer(unittest.TestCase):
    def setUp(self):
        self.ham = ElectronicHamiltonian(np.array([[12500]]),
                                    bath=DebyeBath(CM_K * 77, 35, 106),
                                    dipoles=[[1, 0, 0]])
        self.dyn = RedfieldModel(self.ham, discard_imag_corr=True,
                                 unit_convert=CM_FS)

    def test_absorption_spectra(self):
        f, X = absorption_spectra(self.dyn, 10000)
        self.assertAlmostEqual(f[np.argmax(X)], 12500)
        self.assertTrue(is_constant(np.diff(f)))
        self.assertTrue(np.all(np.diff(X)[:int(len(X) / 2.0 - 1)] > 0))
        self.assertTrue(np.all(np.diff(X)[-int(len(X) / 2.0 - 1):] < 0))

        dyn2 = RedfieldModel(self.ham, rw_freq=12400,
                             discard_imag_corr=True, unit_convert=CM_FS)
        f2, X2 = absorption_spectra(dyn2, 10000)
        self.assertAlmostEqual(f2[np.argmax(X2)], 12500, 0)

class TestRedfieldDimer(unittest.TestCase):
    def setUp(self):
        self.ham = ElectronicHamiltonian(
            np.array([[12881, 120], [120, 12719]]),
            bath=DebyeBath(CM_K * 77, 35, 106),
            dipoles=[[1, 0, 0], [2 * np.cos(.3), 2 * np.sin(.3), 0]])

    def get_model(self, evolve_basis, sparse_matrix):
        return RedfieldModel(self.ham, hilbert_subspace='gef',
                             discard_imag_corr=True, unit_convert=CM_FS,
                             evolve_basis=evolve_basis,
                             sparse_matrix=sparse_matrix)

    def test_evolve_basis_sparse_matrix(self):
        linear_responses = []
        for basis, sp in itertools.product(['site', 'eigen'], [False, True]):
            model = self.get_model(basis, sp)
            f, X = absorption_spectra(model, 10000)
            linear_responses.append(X)

        test_response = linear_responses.pop()
        for resp in linear_responses:
            np.allclose(test_response, resp)
