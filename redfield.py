import numpy as np

from constants import CM_FS
from dynamics import LiouvilleSpaceDynamics
from hamiltonian import transform_out_basis
from operator_tools import tensor_to_super, S_commutator


def redfield_tensor(hamiltonian, subspace='ge', secular=True,
                    discard_imag_corr=False):
    """Calculates the Redfield tensor elements in the energy eigenbasis
    """
    N = hamiltonian.system.n_states(subspace)
    E = hamiltonian.system.E(subspace)

    K = [hamiltonian.system.site_to_exc(coupling, subspace)
         for coupling in hamiltonian.system_bath_coupling(subspace)]
    xi = np.einsum('iab,icd->abcd', K, K)

    if discard_imag_corr:
        corr_func = hamiltonian.bath.corr_func_real
    else:
        corr_func = hamiltonian.bath.corr_func_complex
    corr = np.array([[corr_func(Ei - Ej) for Ej in E] for Ei in E])

    Gamma = np.einsum('abcd,dc->abcd', xi, corr)

    I = np.identity(N)
    Gamma_summed = np.einsum('abbc->ac', Gamma)
    R = (np.einsum('ac,bd->abcd', I, Gamma_summed).conj()
         + np.einsum('bd,ac->abcd', I, Gamma_summed)
         - np.einsum('cabd->abcd', Gamma).conj()
         - np.einsum('dbac->abcd', Gamma))

    if secular:
        R *= secular_terms(N)

    return R


def secular_terms(N):
    I = np.identity(N, dtype=bool)
    return np.einsum('ab,cd->abcd', I, I) | np.einsum('ac,bd->abcd', I, I)


def redfield_dissipator(*args, **kwargs):
    return tensor_to_super(redfield_tensor(*args, **kwargs))


@transform_out_basis(from_basis='exciton')
def redfield_evolve(hamiltonian, subspace='ge', **kwargs):
    H = np.diag(hamiltonian.system.E(subspace))
    R = redfield_dissipator(hamiltonian, subspace, **kwargs)
    return -1j * S_commutator(H) - R


class RedfieldDynamics(LiouvilleSpaceDynamics):
    def __init__(self, hamiltonian, rw_freq=None, subspace=None,
                 restrict_states=None, **kwargs):
        super(RedfieldDynamics, self).__init__(hamiltonian, rw_freq, subspace,
                                               restrict_states)
        self.redfield_super = CM_FS * redfield_evolve(self.hamiltonian,
                                                      subspace=subspace,
                                                      **kwargs)

    def step(self, rho):
        return self.redfield_super.dot(rho)
