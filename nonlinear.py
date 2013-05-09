from __future__ import division
import itertools as it
import numpy as np
import scipy
import scipy.linalg
import scipy.integrate
from numpy import pi, cos, sin, tan, sqrt, log, exp
from numpy.fft import fftshift, ifftshift, fft, fft2
from redfield import L_redfield, operator_1_to_01, operator_1_to_012, \
    cor_ohmic_exp_re
from spectra_tools import density_subset, operator_1_to_012, tensor_to_super, \
    vec,  N_2_from_1, SpectraError, dipole_matrices, S_left, S_right, \
    S_commutator, polarization_setup, density_pop_indices, CM_FS, meta_fft, \
    rho_ee_only, transition_dipole
from utils import MetaArray, memoize, odeint
from pump_probe import default_sample_rate, isotropic_avg_2, simulate_pulse


def greens_function(H_1, t_max, rw_freq=12500, sample_rate=0, piecewise=False,
                    desired_subset='gg,ge,eg,ee,gf,fg,ef,fe,ff', **kargs):
    if sample_rate == 0:
        sample_rate = default_sample_rate(H_1, rw_freq)

    L = CM_FS * L_redfield(H_1, subspace='012', basis='sites',
                           rw_freq=rw_freq, **kargs)

    t_range = np.arange(0, t_max, 1.0/sample_rate)

    if piecewise:
        G = np.empty((len(t_range), L.shape[0], L.shape[1]), dtype=complex)
        for ss_txt in desired_subset.split(','):
            ss = density_subset(ss_txt, H_1.shape[0])
            m_ss = np.ix_(ss, ss)
            G_ss = np.ix_(range(len(t_range)), ss, ss)
            G[G_ss] = np.array([scipy.linalg.expm(L[m_ss] * t)
                                for t in t_range])
    else:
        G = np.array([scipy.linalg.expm(L*t) for t in t_range])

    return MetaArray(G, ticks=t_range, rw_freq=rw_freq)


def fft_greens(G):
    G1 = np.r_[np.zeros_like(G), G]
    G_f = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(G1, axes=0), axis=0),
                          axes=0)
    return G_f


def freqs(t, rw_freq=12500):
    dt = t[1] - t[0]
    n = 2.0*len(t)
    return np.arange(-n/2, n/2, 1)/(3e-5*dt*n) + rw_freq


def nonlinear_response(G_full, dipoles, direction=(-1,1,1),
                       polarization=polarization_setup('xxxx')):

    N_1 = dipoles.shape[0]
    N_2 = N_2_from_1(N_1)

    if direction == (-1,1,1) or direction == (1,-1,1):
        parts = 'gg,eg,ge,ee,fe'
    elif direction == (1,1,-1):
        parts = 'gg,eg,ge,ee,fe,fg'

    ss = density_subset(parts, N_1, vec)
    m_ss = np.ix_(ss, ss)
    m_tss = np.ix_(np.arange(len(G_full)), ss, ss)

    G = G_full[m_tss]

    mu = np.einsum('ijk,lk->lij', dipole_matrices(dipoles), polarization)

    V = np.zeros((4, len(ss), len(ss)), dtype=complex)
    op = {1: np.tril, -1: np.triu}
    for i in range(3):
        V[i] = S_commutator(op[direction[i]](mu[i]))[m_ss]
    V[3] = S_left(op[-1](mu[3]))[m_ss]

    rho0 = np.r_[1, np.zeros(N_2**2 - 1, dtype=complex)][ss]
    tr = vec(np.diag(np.ones(N_2, dtype=complex)))[ss]

    V0_rho0 = V[0].dot(rho0)
    G_V0_rho0 = np.einsum('aij,j', G, V0_rho0)
    V1_G_V0_rho0 = np.einsum('ij,aj', V[1], G_V0_rho0)

    tr_V3 = tr.dot(V[3])
    tr_V3_G = np.einsum('i,aij', tr_V3, G)
    tr_V3_G_V2 = tr_V3_G.dot(V[2])
    tr_V3_G_V2_G = np.einsum('ai,bij', tr_V3_G_V2, G)

    R = np.einsum('abi,ci', tr_V3_G_V2_G, V1_G_V0_rho0)
    return MetaArray(R, ticks=(G.ticks, G.ticks, G.ticks), rw_freq=G.rw_freq)


condition_number = lambda x: norm(x,2)*norm(scipy.linalg.pinv(x),2)

def unique_density(A):
    return np.triu(A).real + np.tril(A, -1).imag


def elem_matrices(d):
    e = np.zeros((d, d, d, d), dtype=float)
    for i in range(d):
        for j in range(d):
            e[i, j, i, j] = 1
    return e

def density_basis(d, identity_separate=False):
    # Thew et al. Qudit Quantum State Tomography. PRA 012303 (2002).
    e = elem_matrices(d)
    if not identity_separate:
        eta = np.zeros((d, d, d), dtype=complex)
        for r in range(d):
            eta[r] = e[r, r]
    else:
        eta = np.zeros((d-1, d, d), dtype=complex)
        for r in range(d-1):
            eta[r] = sqrt(2/((r+1)*(r+2))) * \
                     (np.sum([e[j, j] for j in range(r)], 0)
                      - (r+1) * e[r+1, r+1])
        eta = np.concatenate(([np.identity(d)], eta))
    Theta = []
    beta = []
    for j in range(d):
        for k in range(j+1, d):
            Theta.append(e[j,k] + e[k,j])
            beta.append(-1j*(e[j,k] - e[k,j]))
    return np.concatenate((eta, Theta, beta))


def state_to_density_vec_tri(d):
    e = elem_matrices(d)
    T = np.zeros((d**2, d, d), dtype=complex)
    for i in range(d):
        for j in range(i, d):
            if i == j:
                T[i + d*i] = e[i, i]
            else:
                T[d*i + j] = e[i, j] + e[j, i]
                T[i + d*j] = 1j*(e[i, j] - e[j, i])
    return T.reshape((d**2, d**2), order='F')

def state_to_density(d, *a, **b):
    return np.einsum('rij->ijr', density_basis(d, *a, **b))


def state_to_density_vec(d, *a, **b):
    return np.einsum('rij->ijr', density_basis(d, *a, **b)).reshape(
                     (d**2, d**2),
                     order='F')

def pauli_state_to_density_vec():
    return np.array(np.matrix('1 0 0 1; 0 1 -1j 0; 0 1 1j 0; 1 0 0 -1'))/2

def L_decay(H_1, gamma=50, detune=0, rw_freq=12500):
    N_1 = len(H_1)
    H_x = operator_1_to_012(H_1 - rw_freq * np.identity(N_1))
    H_x[-1,-1] += detune
    E_x, U_x = scipy.linalg.eigh(H_x)
    N_x = len(E_x)

    R_diag = np.zeros(N_x**2, dtype=complex)
    R_diag[density_subset('eg,ge,ef,fe', N_1)] = gamma
    R = np.diag(R_diag)

    U_S = np.kron(U_x, U_x)
    Lsys = -1j*S_commutator(H_x) - U_S.dot(R).dot(U_S.T)
    return Lsys


def pump_probe_projector(H_1, dipoles, t_max,
                         polarization=polarization_setup(['x','x']),
                         signal_include='GSB,ESE,ESA', sep_signals=False,
                         sample_rate=0, basis='sites', secular=True,
                         L=None, cor_func=None, rw_freq=12500):

    if basis != 'sites':
        raise SpectraError('only defined for sites')
    N_1 = dipoles.shape[0]
    ss = memoize(lambda txt: density_subset(txt, N_1))

    X = [np.triu(dipole_matrices(dipoles).dot(p)) for p in polarization]
    V = S_commutator(X[0].T)
    Vf = S_left(X[1])
    tr = vec(np.diag(np.ones(N_2_from_1(N_1), dtype=complex)))
    tr_Vf = tr.dot(Vf)

    if sample_rate == 0:
        sample_rate = default_sample_rate(H_1, rw_freq)

    if L is None:
        L = pi*6e-5 * L_redfield(H_1, cor_func,
                                 subspace='012', basis='sites',
                                 rw_freq=rw_freq, secular=secular)

    t_range = np.arange(0, t_max, 1.0 / sample_rate)

    tr_Vf_eg = tr_Vf[ss('eg')]
    L_eg = L[np.ix_(ss('eg'), ss('eg'))]

    def dtr_Vf_G_eg(t, O):
        return O.dot(L_eg)
    tr_Vf_G_eg = odeint(dtr_Vf_G_eg, tr_Vf_eg, t_range, 'zvode',
                        method='adams', rtol=1e-8)

    tr_Vf_G_V_ee = np.zeros((len(t_range), N_1 ** 2), dtype=complex)

    if 'GSB' in signal_include:
        tr_Vf_G_V_GSB = np.einsum('ni,ij->nj', tr_Vf_G_eg,
                                  V[np.ix_(ss('eg'), ss('gg'))])
        # ground state population is negative of total excited state population
        pop_indices = density_pop_indices(N_1)
        tr_Vf_G_V_ee[:, pop_indices] -= tr_Vf_G_V_GSB

    if 'ESE' in signal_include:
        tr_Vf_G_V_ESE = np.einsum('ni,ij->nj', tr_Vf_G_eg,
                                  V[np.ix_(ss('eg'), ss('ee'))])
        tr_Vf_G_V_ee += tr_Vf_G_V_ESE

    if 'ESA' in signal_include:
        tr_Vf_fe = tr_Vf[ss('fe')]
        L_fe = L[np.ix_(ss('fe'), ss('fe'))]

        def dtr_Vf_G_fe(t, O):
            return O.dot(L_fe)
        tr_Vf_G_fe = odeint(dtr_Vf_G_fe, tr_Vf_fe, t_range, 'zvode',
                            method='adams', rtol=1e-8)
        tr_Vf_G_V_ESA = np.einsum('ni,ij->nj', tr_Vf_G_fe,
                                  V[np.ix_(ss('fe'), ss('ee'))])
        tr_Vf_G_V_ee += tr_Vf_G_V_ESA

    if sep_signals:
        return tr_Vf_G_V_GSB, tr_Vf_G_V_ESE, tr_Vf_G_V_ESA

    return MetaArray(tr_Vf_G_V_ee, ticks=t_range, rw_freq=rw_freq)

pump_probe_projector_iso = isotropic_avg_2(pump_probe_projector)

def pp_4ways(rho0, H_1, t_max, dipoles,
             polarization=polarization_setup(['x','x']),
             sample_rate=0, rw_freq=12500):
    N_1 = dipoles.shape[0]
    ss = memoize(lambda s: density_subset(s, N_1))

    X = [np.triu(dipole_matrices(dipoles).dot(p)) for p in polarization]
    V = S_commutator(X[0].T)[np.ix_(ss('eg'), ss('ee'))]
    Vf = S_left(X[1])
    tr = vec(np.diag(np.ones(N_2_from_1(N_1), dtype=complex)))
    tr_Vf = np.einsum('i,ij', tr, Vf)[ss('eg')]

    V_rho0 = V.dot(rho0)

    if sample_rate == 0:
        sample_rate = default_sample_rate(H_1, rw_freq)

    L = pi*6e-5 * L_redfield(H_1, subspace='012', basis='sites',
                             rw_freq=rw_freq)[np.ix_(ss('eg'), ss('eg'))]

    t = np.arange(0, t_max, 1.0/sample_rate)

    def dtr_Vf(t, tr_Vf):
        return tr_Vf.dot(L)
    tr_Vf_G = odeint(dtr_Vf, tr_Vf, t, 'zvode', method='adams', rtol=1e-8)
    S1 = MetaArray(np.einsum('ni,i->n', tr_Vf_G, V_rho0),
                   ticks=t, rw_freq=rw_freq)

    def drho(t, rho):
        return L.dot(rho)
    G_V_rho0 = odeint(drho, V_rho0, t, 'zvode', method='adams', rtol=1e-8)
    S2 = MetaArray(np.einsum('i,ni->n', tr_Vf, G_V_rho0),
                   ticks=t, rw_freq=rw_freq)

    G_ss = np.ix_(range(len(t)), ss('eg'), ss('eg'))
    G = greens_function(H_1, t_max, piecewise=True, desired_subset='eg')[G_ss]
    S3 = MetaArray(np.einsum('i,nij,j->n', tr_Vf, G, V_rho0),
                   ticks=t, rw_freq=rw_freq)

    tr_Vf_G_V = np.einsum('ni,ij->nj', tr_Vf_G, V)
#    return tr_Vf_G_V
    S4 = MetaArray(np.einsum('ni,i->n', tr_Vf_G_V, rho0),
                   ticks=t, rw_freq=rw_freq)

    return S1, S2, S3, S4
#pump_probe_projector_ge_iso = isotropic_avg(pump_probe_projector_ge)


def pump_probe_from_density(rho0, H_1, t_max, dipoles,
                            polarization=polarization_setup(['x','x']),
                            sample_rate=0, rw_freq=12500):
    N_1 = dipoles.shape[0]
    ss = memoize(lambda txt: density_subset(txt, N_1))

    X = [np.triu(dipole_matrices(dipoles).dot(p)) for p in polarization]
    V = S_commutator(X[0].T)
    Vf = S_left(X[1])
    tr = vec(np.diag(np.ones(N_2_from_1(N_1), dtype=complex)))
    tr_Vf = tr.dot(Vf)

    tr_Vf_ge = tr_Vf[ss('ge')]

    V_rho = V[np.ix_(ss('ge'), ss('ee'))].dot(rho0)

    if sample_rate == 0:
        sample_rate = default_sample_rate(H_1, rw_freq)

    L = pi*6e-5 * L_redfield(H_1, subspace='012', basis='sites',
                             rw_freq=rw_freq)
    L_ge = L[np.ix_(ss('ge'), ss('ge'))]

    t_range = np.arange(0, t_max, 1.0/sample_rate)

    def drho(t, rho):
        return L_ge.dot(rho)

    G_V_rho = odeint(drho, V_rho, t_range, 'zvode', method='adams', rtol=1e-8)

    tr_Vf_G_V = np.einsum('i,ti->t', tr_Vf_ge, G_V_rho)
    return MetaArray(tr_Vf_G_V, ticks=t_range, rw_freq=rw_freq)


def normalized_signal(S):
    return S/np.sqrt(np.sum(np.abs(S.conj().dot(S))))


def combine_rp_nr(Rrp, Rnr, pad_t3=0):
    x3, x2, x1 = Rrp.shape
    y3, y2, y1 = Rnr.shape
    if (x3 != y3) or (x2 != y2):
        raise Exception('Mismatched t3 or t2')
    Rboth = np.zeros((x3 + pad_t3, x2, x1 + y1 - 1), dtype=Rrp.dtype)
    Rboth[pad_t3:, :, :(x1-1)] = Rrp[:, :, :0:-1]
    Rboth[pad_t3:, :, x1:] = Rnr[:, :, 1:]
    Rboth[pad_t3:, :, (x1-1)] = (Rnr[:, :, 0] + Rrp[:, :, 0])/2.0

    t3 = np.concatenate((-Rrp.ticks[0][pad_t3:0:-1], Rrp.ticks[0]))
    t2 = Rrp.ticks[1]
    t1 = np.concatenate((-Rrp.ticks[2][:0:-1], Rnr.ticks[2]))

    return MetaArray(Rboth, ticks=(t3, t2, t1), rw_freq=Rrp.rw_freq)


def response_shift(R0):
    R = R0.copy()
    if R.shape[1] != R.shape[0]:
        raise SpectraError('Matrix not square')
    for i in range(R.shape[0]):
        R[i] = np.roll(R[i], -i, axis=0)
        for j in range(R.shape[1]):
            R[i,j] = np.roll(R[i,j], -(i+j), axis=0)
    return R


def response_enlarge(R0):
    n = R0.shape[0]
    R = np.zeros((n, 2*n, 3*n), dtype=R0.dtype)
    R[:n,:n,:n] = R0
    return R



def response_shift_enlarge(R0):
    n = R0.shape[0]
    R = np.zeros((n, 2*n, 3*n), dtype=R0.dtype)
    R[:n,:n,:n] = R0
    for i in range(n):
        R[i] = np.roll(R[i], -i, axis=0)
        for j in range(n):
            R[i,j] = np.roll(R[i,j], -(i+j), axis=0)
    return R



def inverse_response_shift(R0):
    R = R0.copy()
    if R.shape[1] != R.shape[0]:
        raise SpectraError('Matrix not square')
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            R[i,j] = np.roll(R[i,j], (i+j), axis=0)
        R[i] = np.roll(R[i], i, axis=0)
    return R


def nonlinear_response_2D(G_full, dipoles, direction=(-1,1,1),
                          polarization=polarization_setup('xxxx')):

    N_1 = dipoles.shape[0]
    N_2 = N_2_from_1(N_1)

    if direction == (-1,1,1) or direction == (1,-1,1):
        parts = 'gg,eg,ge,ee,fe'
    elif direction == (1,1,-1):
        parts = 'gg,eg,ge,ee,fe,fg'

    ss = density_subset(parts, N_1, vec)
    m_ss = np.ix_(ss, ss)
    m_tss = np.ix_(np.arange(len(G_full)), ss, ss)

    G = G_full[m_tss]
    G_f = fft_greens(G)

    mu = np.einsum('ijk,lk->lij', dipole_matrices(dipoles), polarization)

    V = np.zeros((4, len(ss), len(ss)), dtype=complex)
    op = {1: np.tril, -1: np.triu}
    for i in range(3):
        V[i] = S_commutator(op[direction[i]](mu[i]))[m_ss]
    V[3] = S_left(np.triu(mu[3]))[m_ss]

    rho0 = np.r_[1, np.zeros(N_2**2 - 1, dtype=complex)][ss]
    tr = vec(np.diag(np.ones(N_2, dtype=complex)))[ss]

    V0_rho0 = V[0].dot(rho0)
    Gf_V0_rho0 = np.einsum('aij,j', G_f, V0_rho0)
    V1_Gf_V0_rho0 = np.einsum('ij,aj', V[1], Gf_V0_rho0)

    tr_V3 = tr.dot(V[3])
    tr_V3_Gf = np.einsum('i,aij', tr_V3, G_f)
    tr_V3_Gf_V2 = tr_V3_Gf.dot(V[2])
    tr_V3_Gf_V2_G = np.einsum('ai,bij', tr_V3_Gf_V2, G)

    R = np.einsum('abi,ci', tr_V3_Gf_V2_G, V1_Gf_V0_rho0)

    return R


def pump_impulsive_probe(H, dipoles, freq_bounds=(12000, 12800),
                         pump_args=None, probe_args=None):
    if pump_args is None:
        pump_args = {}
    if probe_args is None:
        probe_args = {}
    rho = isotropic_avg_2(simulate_pulse)(H, dipoles, **pump_args)
    Pfft = meta_fft(-pump_probe_projector_iso(H, dipoles, **probe_args),
                    freq_bounds=freq_bounds)
    Rpp = MetaArray(np.einsum('fi,ti->ft', Pfft, rho_ee_only(rho)),
                    ticks=[Pfft.ticks, rho.ticks])
    return Rpp
