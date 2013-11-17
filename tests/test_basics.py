import numpy as np
import unittest

from qspectra import (ElectronicHamiltonian, DebyeBath, CM_K, CM_FS,
                      RedfieldModel, absorption_spectra)
from qspectra.simulate.utils import is_constant


class TestMonomer(unittest.TestCase):
    def setUp(self):
        ham = ElectronicHamiltonian(np.array([[12500]]),
                                    bath=DebyeBath(CM_K * 77, 35, 106),
                                    dipoles=[[1, 0, 0]])
        self.dyn = RedfieldModel(ham, hilbert_subspace='ge',
                                 discard_imag_corr=True, unit_convert=CM_FS)

    def test_absorption_spectra(self):
        f, X = absorption_spectra(self.dyn, 10000)
        self.assertAlmostEqual(f[np.argmax(X)], 12500)
        self.assertTrue(is_constant(np.diff(f)))
        self.assertTrue(np.all(np.diff(X)[:int(len(X) / 2.0 - 1)] > 0))
        self.assertTrue(np.all(np.diff(X)[-int(len(X) / 2.0 - 1):] < 0))
