import itertools
import numpy as np

from constants import CM_FS
from operator_tools import (diag_vec, density_subset, density_pop_indices,
                            exclude_ground, meta_series_op,
                            normalized_exclude_ground)
from redfield import redfield_evolve
from simulate import simulate_pump
from spectra_tools import (isotropic_average_2nd_order, meta_fft)
from utils import odeint, MetaArray, memoize


def impulsive_probe(dynamics, rho_pump, polarization='xx', t_max=5000):
    t = np.arange(0, t_max, dynamics.time_step)
    S = MetaArray(np.zeros(t.shape, complex), ticks=t, rw_freq=dynamics.rw_freq)
    for states in ('eg', 'fe'):
        dyn = dynamics.restrict_states(states)
        V2 = dyn.dipole_create(polarization[0])
        V3 = dyn.dipole_destroy(polarization[1])
        V2_rho = V2.commutator(rho_pump)
        G_V2_rho = dyn.integrate(initial_state=V2_rho, t=t)
        S += -1j * V3.measure(G_V2_rho)
    return S


def aniso_pairs(items):
    for i in items:
        for j in items:
            yield i + j, i + j
            yield i + j, j + i


def simulate_pump_probe_iso(hamiltonian, pump, t_extra_pump, t_max_probe,
                            rw_freq=None, freq_bounds=None, normalize=False,
                            append_thermal=False, **kwargs):
    rho_pump = 3 * isotropic_average_2nd_order(simulate_pump)(
        hamiltonian, t_extra_pump, pump=pump, rw_freq=rw_freq,
        append_thermal=append_thermal, **kwargs)
    proj = 3 * isotropic_average_2nd_order(pump_probe_projector)(
        hamiltonian, t_max_probe, rw_freq=rw_freq, **kwargs)
    proj_fft = meta_fft(-proj, freq_bounds=freq_bounds)

    exc_ground = (normalized_exclude_ground if normalize else
                  meta_series_op(exclude_ground))
    return MetaArray(np.einsum('fi,ti->tf', proj_fft, exc_ground(rho_pump)),
                     ticks=[rho_pump.ticks, proj_fft.ticks])


def simulate_pump_probe_aniso(hamiltonian, pump, t_extra_pump,
                              t_max_probe, rw_freq=None, freq_bounds=None,
                              append_thermal=False, **kwargs):
    rho_pump = {}
    proj_fft = {}
    for (i, j) in itertools.product('xyz', 'xyz'):
        polar = i + j
        rho_pump[polar] = simulate_pump(hamiltonian, t_extra_pump, polar,
                                        pump, rw_freq=rw_freq,
                                        append_thermal=append_thermal, **kwargs)
        rho_ticks = rho_pump[polar].ticks
        proj = pump_probe_projector(hamiltonian, t_max_probe, polar,
                                    rw_freq=rw_freq, **kwargs)
        proj_fft[polar] = meta_fft(-proj, freq_bounds=freq_bounds)
        proj_fft_ticks = proj_fft[polar].ticks

    exc_ground = meta_series_op(exclude_ground)
    return sum(MetaArray(np.einsum('fi,ti->tf', proj_fft[polar_pr],
                                   exc_ground(rho_pump[polar_pu])),
                         ticks=[rho_ticks, proj_fft_ticks])
               for polar_pu, polar_pr in aniso_pairs('xyz'))


def pump_probe_projector(hamiltonian, t_max, polarization='xx', rw_freq=None,
                         signal_include='GSB,ESE,ESA',
                         secular=True, discard_imag_corr=False,
                         ode_settings=dict(rtol=1e-8, method='adams')):

    ss = memoize(lambda subset:
                 density_subset(subset, hamiltonian.system.n_sites))

    V = hamiltonian.dipole_destroy_evolve(polarization[0], basis='site',
                                          subspace='gef')
    Vf = 1j * hamiltonian.dipole_destroy_left(polarization[1], basis='site',
                                              subspace='gef')

    tr = diag_vec(hamiltonian.system.n_states('gef'))
    tr_Vf = tr.dot(Vf)

    N = hamiltonian.system.n_sites
    if rw_freq is None:
        rw_freq = hamiltonian.system.central_freq
    hamiltonian = hamiltonian.to_rotating_frame(rw_freq)
    Lsys = CM_FS * redfield_evolve(hamiltonian, basis='site',
                                   subspace='gef', secular=secular,
                                   discard_imag_corr=discard_imag_corr)

    t_range = np.arange(0, t_max, hamiltonian.time_step)

    tr_Vf_eg = tr_Vf[ss('eg')]
    L_eg = Lsys[np.ix_(ss('eg'), ss('eg'))]

    def dtr_Vf_G_eg(O, t):
        return O.dot(L_eg)
    tr_Vf_G_eg = odeint(dtr_Vf_G_eg, tr_Vf_eg, t_range, 'zvode',
                        **ode_settings)

    tr_Vf_G_V_ee = np.zeros((len(t_range), N ** 2), dtype=complex)

    if 'GSB' in signal_include:
        tr_Vf_G_V_GSB = np.einsum('ni,ij->nj', tr_Vf_G_eg,
                                  V.T[np.ix_(ss('eg'), ss('gg'))])
        # ground state population is negative of total excited state population
        pop_indices = density_pop_indices(N)
        tr_Vf_G_V_ee[:, pop_indices] -= tr_Vf_G_V_GSB

    if 'ESE' in signal_include:
        tr_Vf_G_V_ESE = np.einsum('ni,ij->nj', tr_Vf_G_eg,
                                  V.T[np.ix_(ss('eg'), ss('ee'))])
        tr_Vf_G_V_ee += tr_Vf_G_V_ESE

    if 'ESA' in signal_include:
        tr_Vf_fe = tr_Vf[ss('fe')]
        L_fe = Lsys[np.ix_(ss('fe'), ss('fe'))]

        def dtr_Vf_G_fe(O, t):
            return O.dot(L_fe)
        tr_Vf_G_fe = odeint(dtr_Vf_G_fe, tr_Vf_fe, t_range, 'zvode',
                            **ode_settings)
        tr_Vf_G_V_ESA = np.einsum('ni,ij->nj', tr_Vf_G_fe,
                                  V.T[np.ix_(ss('fe'), ss('ee'))])
        tr_Vf_G_V_ee += tr_Vf_G_V_ESA

    return MetaArray(tr_Vf_G_V_ee, ticks=t_range, rw_freq=rw_freq,
                     GSB=tr_Vf_G_V_GSB, ESE=tr_Vf_G_V_ESE, ESA=tr_Vf_G_V_ESA)
