import numpy as np

from ..operator_tools import basis_transform_operator
from .liouville_space import (super_commutator_matrix, tensor_to_super,
                              LiouvilleSpaceModel)
from ..utils import memoized_property


def redfield_tensor(hamiltonian, subspace='ge', secular=True,
                    discard_imag_corr=False):
    """
    Calculates the Redfield tensor elements as a 4D array in the energy
    eigenbasis

    All baths are assumed to be uncorrelated and of identical form.

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
    May and Kuhn (2011), Charge and Energy Transfer Dynamics in Molecular
    Systems, Third Edition.
    """
    n_states = hamiltonian.n_states(subspace)
    energies = hamiltonian.E(subspace)

    K = [basis_transform_operator(coupling, hamiltonian.U(subspace))
         for coupling in hamiltonian.system_bath_couplings(subspace)]

    if discard_imag_corr:
        corr_func = hamiltonian.bath.corr_func_real
    else:
        corr_func = hamiltonian.bath.corr_func_complex
    # this is the "one-sided correlation function" as a function of frequency,
    # C(\omega) = \int_0^\infty \exp(i \omega t) C(t) dt
    C = np.array([[corr_func(Ei - Ej) for Ej in energies] for Ei in energies])

    # May and Kuhn, Eq (3.319)
    # The assumption of uncorrelated baths (i.e., $C_{uv}(t) = 0$) allows us to
    # drop the index v. However, unlike May and Kuhn, we do not necessarily
    # discard the imaginary part of this variable, depending on the argument
    # discard_imag_corr as used above above for determining C (this does
    # implicitly assume that all system-bath coupling matrices are real).
    Gamma = np.einsum('iab,icd,dc->abcd', K, K, C)

    # May and Kuhn, Eq (3.322)
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


def redfield_evolve(hamiltonian, subspace='ge', evolve_basis='site', **kwargs):
    H = np.diag(hamiltonian.E(subspace))
    R = redfield_dissipator(hamiltonian, subspace, **kwargs)
    L = -1j * super_commutator_matrix(H) - R
    if evolve_basis == 'site':
        return basis_transform_operator(L, hamiltonian.U(subspace).T.conj())
    elif evolve_basis == 'eigen':
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
                 unit_convert=1, secular=True, discard_imag_corr=False,
                 evolve_basis='site', sparse_matrix=False):
        super(RedfieldModel, self).__init__(hamiltonian, rw_freq,
                                            hilbert_subspace, unit_convert,
                                            evolve_basis, sparse_matrix)
        self.secular = secular
        self.discard_imag_corr = discard_imag_corr

    @memoized_property
    def evolution_super_operator(self):
        return (self.unit_convert
                * redfield_evolve(self.hamiltonian, self.hilbert_subspace,
                                  evolve_basis=self.evolve_basis,
                                  secular=self.secular,
                                  discard_imag_corr=self.discard_imag_corr))
