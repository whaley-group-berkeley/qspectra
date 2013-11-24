import numpy as np
import unittest

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
