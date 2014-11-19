import numpy as np

from ..operator_tools import basis_transform
from .liouville_space import (super_commutator_matrix, tensor_to_super,
                              LiouvilleSpaceModel)
from ..utils import memoized_property


def redfield_tensor(hamiltonian, subspace='ge', secular=True,
                    discard_imag_corr=False):
    """
    Calculates the Redfield tensor elements as a 4D array in the energy
    eigenbasis

    Each bath is assumed to be of identical form

    Parameters
    ----------
    hamiltonian : hamiltonian.Hamiltonian
        Hamiltonian object specifying the system
    subspace : container, default 'ge'
        Container of any or all of 'g', 'e' and 'f' indicating the desired
        subspaces on which to calculate the Redfield tensor
    secular : boolean, default True
        Whether to employ the secular approximation and Bloch model to neglect
        all terms other than coherence decay and population transfer
    discard_imag_corr : boolean, default False
        Whether to discard the imaginary part of the bath correlation functions

    Returns
    -------
    out : np.ndarray
        Four dimensional array given the Redfield transfer rates between
        density matrix elements in the system energy eigenbasis

    References
    ----------
    Nitzan
    """
    n_states = hamiltonian.n_states(subspace)
    energies = hamiltonian.E(subspace)

    K = [basis_transform(coupling, hamiltonian.U(subspace))
         for coupling in hamiltonian.system_bath_couplings(subspace)]
    xi = np.einsum('iab,icd->abcd', K, K)

    if discard_imag_corr:
        corr_func = hamiltonian.bath.corr_func_real
    else:
        corr_func = hamiltonian.bath.corr_func_complex
    corr = np.array([[corr_func(Ei - Ej) for Ej in energies]
                     for Ei in energies])

    Gamma = np.einsum('abcd,dc->abcd', xi, corr)

    I = np.identity(n_states)
    Gamma_summed = np.einsum('abbc->ac', Gamma)
    R = (np.einsum('ac,bd->abcd', I, Gamma_summed).conj()
         + np.einsum('bd,ac->abcd', I, Gamma_summed)
         - np.einsum('cabd->abcd', Gamma).conj()
         - np.einsum('dbac->abcd', Gamma))

    if secular:
        R *= secular_terms(n_states)

    return R


def secular_terms(n_states):
    """
    Returns a boolean array of all terms that survive the secular/Bloch
    approximation
    """
    I = np.identity(n_states, dtype=bool)
    return np.einsum('ab,cd->abcd', I, I) | np.einsum('ac,bd->abcd', I, I)


def redfield_dissipator(*args, **kwargs):
    """
    Returns a super-operator representation the Redfield dissipation tensor

    Arguments are passed to the redfield_tensor function
    """
    return tensor_to_super(redfield_tensor(*args, **kwargs))


def redfield_evolve(hamiltonian, subspace='ge', evolve_basis='exciton', **kwargs):
    H = np.diag(hamiltonian.E(subspace))
    R = redfield_dissipator(hamiltonian, subspace, **kwargs)
    L = -1j * super_commutator_matrix(H) - R
    if evolve_basis == 'site':
        return basis_transform(L, hamiltonian.U(subspace).T.conj())
    elif evolve_basis == 'exciton':
        return L
    else:
        raise ValueError('invalid basis')


class RedfieldModel(LiouvilleSpaceModel):
    """
    DynamicalModel for Redfield theory

    Assumes that each pigment is coupled to an identical, independent bath

    Parameters
    ----------
    hamiltonian : hamiltonian.Hamiltonian
        Hamiltonian object specifying the system
    rw_freq : float, optional
        Rotating wave frequency at which to calculate dynamics. By default,
        the rotating wave frequency is chosen from the central frequency
        of the Hamiltonian.
    hilbert_subspace : container, default 'ge'
        Container of any or all of 'g', 'e' and 'f' indicating the desired
        Hilbert subspace on which to calculate the Redfield tensor.
    unit_convert : number, optional
        Unit conversion from energy to time units (default 1).
    secular : boolean, default True
        Whether to employ the secular approximation and Bloch model to
        neglect all terms other than coherence decay and population transfer
    discard_imag_corr : boolean, default False
        Whether to discard the imaginary part of the bath correlation
        functions

    References
    ----------
    .. [1] Nitzan (2006)
    """
    def __init__(self, hamiltonian, rw_freq=None, hilbert_subspace='gef',
                 unit_convert=1, secular=True, discard_imag_corr=False):
        super(RedfieldModel, self).__init__(hamiltonian, rw_freq,
                                            hilbert_subspace, unit_convert)
        self.secular = secular
        self.discard_imag_corr = discard_imag_corr

    @memoized_property
    def evolution_super_operator(self):
        return (self.unit_convert
                * redfield_evolve(self.hamiltonian, self.hilbert_subspace,
                                  evolve_basis=self.evolve_basis,
                                  secular=self.secular,
                                  discard_imag_corr=self.discard_imag_corr))
