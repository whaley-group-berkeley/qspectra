"""
Calculate the linear response
"""

from __future__ import division
import numpy as np
import scipy
import scipy.linalg
import scipy.integrate
from numpy import pi, cos, sin, tan, sqrt, log, exp
from redfield import redfield_tensor, operator_1_to_012, \
L_redfield, operator_1_to_01, cor_debye
from spectra_tools import density_subset, polarization_setup, CM_FS, CM_K, \
    transition_dipole
from utils import odeint, MetaArray
from pump_probe import default_sample_rate


def linear_response_rw(H_1, dipoles, t_max, cor_func=None, rw_freq=12500,
                       sample_rate=None, secular=True, ode_settings=None):

    N_1 = H_1.shape[0]
    _, U_1 = scipy.linalg.eigh(H_1)

    ss = lambda s: density_subset(s, N_1)

    if cor_func is None:
        cor_func = lambda x: cor_debye(x, T=(CM_K * 77), lmbda=35., gamma=106)

    Lsys = CM_FS * L_redfield(H_1, cor_func, basis='sites',
                              subspace='01', K2=None,
                              secular=secular, rw_freq=rw_freq
                              )[np.ix_(ss('eg'), ss('eg'))]

    if sample_rate is None:
        sample_rate = default_sample_rate(H_1, rw_freq)

    if ode_settings is None:
        ode_settings = {
            'rtol': 1e-8,
            'max_step': 3,
            'nsteps': 1e4
        }

    t = np.arange(0, t_max, 1.0 / sample_rate)
    S = np.zeros(len(t), dtype=complex)

    def drho(t, rho):
        return Lsys.dot(rho)

    for polar in np.eye(3):
        mu = dipoles.dot(np.array(polar, dtype=complex))
        S += odeint(drho, mu, t, 'zvode', **ode_settings).dot(mu)

    return MetaArray(S, ticks=t, rw_freq=rw_freq)
