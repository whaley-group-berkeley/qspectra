import numpy as np
import scipy

from ..operator_tools import basis_transform_operator
from .liouville_space import (super_commutator_matrix, tensor_to_super,
                              LiouvilleSpaceModel)
from ..utils import memoized_property
from qspectra import *

def lindblad_tensor(hamiltonian, avg_lifetime=None, subspace='e',
                    overlap_scale=1):
    """
    Lindblad tensor where the system coupling operators are the
    transfer operators of the form: |i><j|

    """
    n_states = hamiltonian.n_states(subspace)
    energies = hamiltonian.E(subspace)

    R = np.zeros((n_states, n_states, n_states, n_states))

    conj_sq_h = hamiltonian.U(subspace).conj() * hamiltonian.U(subspace)

    lambda_mat = np.zeros((n_states, n_states))
    inverse_lifetime = 1 / avg_lifetime

    T = hamiltonian.bath.temperature

    # skip diagonals since they should stay zero: no pure dephasing
    for i in xrange(n_states):
        for j in xrange(i + 1, n_states):
            # E_i <= E_j
            overlap = 1 - overlap_scale * 0.5 * np.sum(np.abs(conj_sq_h[:,i]
                 - conj_sq_h[:,j]))

            lambda_mat[i,j] = overlap
            lambda_mat[j,i] = overlap * np.exp(
                              (energies[i] - energies[j]) / (CM_K * T))

    norm_const = np.sum(lambda_mat)
    lambda_mat *= inverse_lifetime * n_states / norm_const
    lambda_summed = np.sum(lambda_mat, axis=0)

    for b in xrange(n_states):
        for a in xrange(n_states):
            if a == b:
                R[a,a,a,a] = - lambda_summed[a] + lambda_mat[a,a]
            else:
                R[a,a,b,b] = lambda_mat[a,b]
                R[a,b,a,b] = - 1 / 2 * (lambda_summed[a] + lambda_summed[b])
    return R

def lindblad_superoperator(*args, **kwargs):
    """
    Returns a super-operator representation the Lindblad dissipation tensor

    Arguments are passed to the redfield_tensor function
    """
    return tensor_to_super(lindblad_tensor(*args, **kwargs))

def add_imaginary_term(hamiltonian, site, subspace, magnitude=0.00):
    """ adds a small imaginary term to hamiltonian[site, site]
    returns the hamiltonian in the eignestate basis """
    non_hermitian_H = 1j * np.copy(hamiltonian.H(subspace))
    non_hermitian_H *= 1j
    non_hermitian_H[site, site] = 1j * non_hermitian_H[site, site] * magnitude

    E, U = scipy.linalg.eig(non_hermitian_H)
    print E
    return np.diag(E)

def lindblad_evolve(hamiltonian, subspace='e', add_imag_term=None,
                    evolve_basis='site', **kwargs):
    H = np.diag(hamiltonian.E(subspace))
    R = lindblad_superoperator(hamiltonian, subspace=subspace, **kwargs)
    if add_imag_term is not None:
        H = add_imaginary_term(hamiltonian, self.add_imag_term, subspace)
    L = -1j * super_commutator_matrix(H) + R

    if evolve_basis == 'site':
        return basis_transform_operator(L, hamiltonian.U(subspace).T.conj())
    elif evolve_basis == 'eigen':
        return L
    else:
        raise ValueError('invalid basis')

class LindbladModel(LiouvilleSpaceModel):
    """
    DynamicalModel for Lindblad dynamics

    add_imag_term adds a small imaginary term to the requested site.
    """
    def __init__(self, hamiltonian, rw_freq=None, hilbert_subspace='e',
                 unit_convert=1, add_imag_term=None,
                 avg_lifetime=1, overlap_scale=1,
                 evolve_basis='site', sparse_matrix=False):
        super(LindbladModel, self).__init__(hamiltonian, rw_freq,
                                            hilbert_subspace, unit_convert,
                                            evolve_basis, sparse_matrix)
        self.add_imag_term = add_imag_term
        self.avg_lifetime = avg_lifetime
        self.overlap_scale = overlap_scale

    @memoized_property
    def evolution_super_operator(self):
        return (self.unit_convert
                * lindblad_evolve(self.hamiltonian, self.hilbert_subspace,
                                  evolve_basis=self.evolve_basis,
                                  add_imag_term=self.add_imag_term,
                                  avg_lifetime=self.avg_lifetime,
                                  overlap_scale=self.overlap_scale))
