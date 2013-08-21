import numpy as np

from .generic import DynamicalModel
from ..operator_tools import basis_transform
from .liouville_space import (den_to_vec, extract_subspace,
                              super_commutator_matrix, tensor_to_super,
                              liouville_subspace_indices, LiouvilleSpaceOperator)
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
    n_states = hamiltonian.system.n_states(subspace)
    energies = hamiltonian.system.E(subspace)

    K = [basis_transform(coupling, hamiltonian.system.U(subspace))
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


def redfield_evolve(hamiltonian, subspace='ge', basis='site', **kwargs):
    H = np.diag(hamiltonian.system.E(subspace))
    R = redfield_dissipator(hamiltonian, subspace, **kwargs)
    L = -1j * super_commutator_matrix(H) - R
    if basis == 'site':
        return basis_transform(L, hamiltonian.system.U(subspace).T.conj())
    elif basis == 'exciton':
        return L
    else:
        raise ValueError('invalid basis')


class RedfieldModel(DynamicalModel):
    def __init__(self, hamiltonian, rw_freq=None, subspace='gef',
                 unit_convert=1, **redfield_evolve_kwargs):
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
        subspace : container, default 'ge'
            Container of any or all of 'g', 'e' and 'f' indicating the desired
            subspaces on which to calculate the Redfield tensor.
        unit_convert : number, optional
            Unit conversion from energy to time units (default 1).
        basis : 'site' or 'exciton', optional
            Basis in which to simulate dynamics (default 'site')
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
        Nitzan (2006)
        """
        self.hamiltonian = hamiltonian.in_rotating_frame(rw_freq)
        self.rw_freq = self.hamiltonian.system.energy_offset
        self.subspace = subspace
        self.unit_convert = unit_convert
        self.redfield_evolve_kwargs = redfield_evolve_kwargs

    def ground_state(self, liouville_subspace):
        """
        Return the ground state in the given Liouville subspace
        """
        psi0 = self.hamiltonian.system.ground_state(self.subspace)
        rho0 = psi0.conj().reshape(1, -1) * psi0.reshape(-1, 1)
        index = liouville_subspace_indices(liouville_subspace, self.subspace,
                                           self.hamiltonian.system.n_sites)
        return den_to_vec(rho0)[index]

    @memoized_property
    def redfield_super_operator(self):
        return (self.unit_convert
                * redfield_evolve(self.hamiltonian, self.subspace,
                                  **self.redfield_evolve_kwargs))

    def equation_of_motion(self, liouville_subspace):
        """
        Return the equation of motion for this dynamical model in the given
        subspace, a function which takes a state vector and returns its first
        time derivative, for use in a numerical integration routine
        """
        index = liouville_subspace_indices(liouville_subspace, self.subspace,
                                           self.hamiltonian.system.n_sites)
        mesh = np.ix_(index, index)
        evolve = self.redfield_super_operator[mesh]
        def eom(t, rho):
            return evolve.dot(rho)
        return eom

    def dipole_operator(self, liouv_subspace_map, polarization, include_transitions):
        operator = self.hamiltonian.dipole_operator(self.subspace, polarization,
                                                    include_transitions)
        return LiouvilleSpaceOperator(operator, self.subspace, liouv_subspace_map)

    def dipole_destroy(self, liouville_subspace_map, polarization):
        """
        Return a dipole annhilation operator that follows the SystemOperator API
        for the given subspace and polarization
        """
        return self.dipole_operator(liouville_subspace_map, polarization, '-')

    def dipole_create(self, liouville_subspace_map, polarization):
        """
        Return a dipole creation operator that follows the SystemOperator
        API for the given liouville_subspace_map and polarization
        """
        return self.dipole_operator(liouville_subspace_map, polarization, '+')

    @property
    def time_step(self):
        """
        The default time step at which to sample the equation of motion (in the
        rotating frame)
        """
        return self.hamiltonian.time_step / self.unit_convert
