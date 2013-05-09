"""
Implements the method for calculating non-linear spectra as described in these
papers:
1.  Gelin, M. F., Egorova, D. & Domcke, W. Efficient Calculation of Time- and
    Frequency- Resolved Four-Wave-Mixing Signals. Acc. Chem. Res. 42, (2009).
2.  Egorova, D., Gelin, M. F. & Domcke, W. Analysis of cross peaks in two-
    dimensional electronic photon-echo spectroscopy for simple models with
    vibrations and dissipation. J. Chem. Phys. 126, 74314 (2007).
3.  Gelin, M. F., Egorova, D. & Domcke, W. Efficient method for the calculation
    of time- and frequency-resolved four-wave mixing signals and its
    application to photon-echo spectroscopy. J. Chem. Phys. 123, 164112 (2005).
"""
from numpy import conj
import numpy as np

from constants import CM_FS
from operator_tools import unit_vec, density_subset, diag_vec
from redfield import redfield_evolve
import utils


def simulate_pump_probe(hamiltonian, pump, probe, **kwargs):
    control_fields = [pump, pump, probe]
    return simulate_photon_echo(hamiltonian, control_fields, **kwargs)


def simulate_photon_echo(hamiltonian, control_fields, t_extra=1000,
                         polarization='xxxx',
                         rw_freq=None, basis='exciton',
                         subset='gg,eg,ge,ee,fe,fg',
                         secular=True, sparse=False, discard_imag_corr=False,
                         ode_settings=dict(rtol=1e-8, atol=1e-12, max_step=3)):
    """
    Note: 'fg' in subset only necessary if pulse overlap is allowed between
    the 2nd and 3rd pulses
    """
    ss = density_subset(subset, hamiltonian.system.n_sites)
    m_ss = np.ix_(ss, ss)

    op_kwargs = dict(basis=basis, subspace='gef', sparse=sparse)

    if rw_freq is None:
        rw_freq = hamiltonian.system.central_freq
    hamiltonian = hamiltonian.to_rotating_frame(rw_freq)
    Lsys = CM_FS * redfield_evolve(hamiltonian, secular=secular,
                                   discard_imag_corr=discard_imag_corr,
                                   **op_kwargs)[m_ss]

    V = [hamiltonian.dipole_destroy_evolve(polar, **op_kwargs)[m_ss]
         for polar in polarization[:3]]
    Vf = hamiltonian.dipole_destroy_left(polarization[-1],
                                         **op_kwargs)[m_ss]

    rho0 = unit_vec(0, hamiltonian.system.n_states('gef'))[ss]

    E = [lambda t: field(t, rw_freq) for field in control_fields]

    def drho1(rho, t):
        return (Lsys.dot(rho)
                - E[0](t) * V[0].dot(rho)
                - conj(E[1](t)) * V[1].T.dot(rho)
                - conj(E[2](t)) * V[2].T.dot(rho))

    def drho2(rho, t):
        return (Lsys.dot(rho)
                - E[0](t) * V[0].dot(rho)
                - conj(E[1](t)) * V[1].T.dot(rho))

    def drho3(rho, t):
        return (Lsys.dot(rho)
                - E[0](t) * V[0].dot(rho)
                - conj(E[2](t)) * V[2].T.dot(rho))

    t = np.arange(min(Ei.t_init for Ei in E),
                  max(Ei.t_final for Ei in E) + t_extra,
                  hamiltonian.time_step)

    rho1 = utils.odeint(drho1, rho0, t, **ode_settings)
    rho2 = utils.odeint(drho2, rho0, t, **ode_settings)
    rho3 = utils.odeint(drho3, rho0, t, **ode_settings)

    tr = diag_vec(hamiltonian.system.n_states('gef'))[ss]
    tr_Vf = np.einsum('i,ij', tr, Vf)
    S_t = np.einsum('j,tj', tr_Vf, rho1 - rho2 - rho3)

    # rate = hamiltonian.freq_step
    # t_pad = np.arange(t[0] + np.ceil((t[-1] - t[0]) * rate) / rate,
    #                   12000, 1.0 / rate)
    # S_t_pad = np.zeros_like(t_pad)

    # t = np.r_[t, t_pad]
    # S_t = np.r_[S_t, S_t_pad]

    return utils.MetaArray(S_t, ticks=t, rw_freq=rw_freq, pulses=control_fields)
