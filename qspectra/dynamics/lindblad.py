import numpy as np
import scipy

from ..operator_tools import basis_transform
from .liouville_space import (super_commutator_matrix, tensor_to_super,
                              LiouvilleSpaceModel)
from ..utils import memoized_property

class LindbladModel(LiouvilleSpaceModel):
    """
    add_imag_term

    """
    def __init__(self, hamiltonian, rw_freq=None, hilbert_subspace='e',
                 unit_convert=1, add_imag_term=None):
        """
        add_imag_term adds a small imaginary term to the requested site.
        """
        self.add_imag_term = add_imag_term
        self.avg_lifetime = .005
        super(LindbladModel, self).__init__(hamiltonian, rw_freq, hilbert_subspace, unit_convert)

    @memoized_property
    def evolution_super_operator(self):
        return (self.unit_convert * self.lindblad_evolve(self.hamiltonian, self.hilbert_subspace, basis='site'))

    def lindblad_evolve(self, hamiltonian, subspace='e', basis='site', **kwargs):
        H = np.diag(hamiltonian.E(subspace))
        R = self.lindblad_superoperator(hamiltonian, subspace=subspace, **kwargs)

        if self.add_imag_term != None:
            Hp = add_imaginary_term(hamiltonian, self.add_imag_term, subspace)
            S = super_commutator_matrix(Hp)
        else:
            S = super_commutator_matrix(H)

        L = -1j * S + R

        if basis == 'site':
            return basis_transform(L, hamiltonian.U(subspace).T.conj())
        elif basis == 'exciton':
            return L
        else:
            raise ValueError('invalid basis')

    def lindblad_superoperator(self, *args, **kwargs):
        """
        Returns a super-operator representation
        """
        return tensor_to_super(self.lindblad_tensor(*args, **kwargs))

    def lindblad_tensor(self, hamiltonian,  T=300, subspace='e'):
        """
        Lindblad tensor where the system coupling operators are the transfer operators
        of the form: |i><j|
        """
        n_states = hamiltonian.n_states(subspace)
        energies = hamiltonian.E(subspace)

        # matrix of lindblad operators:
        lindblad_ops = []
        R = np.zeros((n_states, n_states, n_states, n_states))

        conj_sq_h = hamiltonian.U(subspace).conj() * hamiltonian.U(subspace)

        lambda_mat = np.zeros((n_states, n_states))
        KB_cm = 0.0695034
        inverse_lifetime = 1 / self.avg_lifetime

        # skip diagonals since they should stay zero: no pure dephasing
        for i in xrange(n_states):
            for j in xrange(i + 1, n_states):
                # E_i <= E_j
                overlap = 1 - 0.5 * np.sum(np.abs(conj_sq_h[:,i] - conj_sq_h[:,j]))
                lambda_mat[i,j] = overlap
                lambda_mat[j,i] = overlap * np.exp((energies[i] - energies[j]) / (KB_cm * T))

        # lambda_mat *= n_states / norm_const
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
        print 'num zero: {}'.format(len(np.where(R == 0)[0]))
        print 'num non zero: {}'.format(len(np.where(R != 0)[0]))
        return R

def add_imaginary_term(hamiltonian, site, subspace, magnitude=0.00):
    """ adds a small imaginary term to hamiltonian[site, site]
    returns the hamiltonian in the eignestate basis """
    non_hermitian_H = 1j * np.copy(hamiltonian.H(subspace))
    non_hermitian_H *= 1j
    non_hermitian_H[site, site] = 1j * non_hermitian_H[site, site] * magnitude

    E, U = scipy.linalg.eig(non_hermitian_H)
    print E
    return np.diag(E)


