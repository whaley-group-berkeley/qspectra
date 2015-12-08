import numpy as np
# import scipy as sp
from scipy import constants
from scipy.sparse import lil_matrix, csr_matrix

from .base import DynamicalModel, SystemOperator
from .liouville_space import (LiouvilleSpaceModel, LiouvilleSpaceOperator,
                              super_commutator_sparse_matrix,
                              super_left_sparse_matrix,
                              super_right_sparse_matrix)
from ..utils import memoized_property

class HEOMSpaceOperator(SystemOperator):
    def __init__(self, dynamical_model, *args, **kwargs):
        self.dynamical_model = dynamical_model
        self.lspace_op = LiouvilleSpaceOperator(*args, **kwargs)
        self.ado_slices = dynamical_model.ado_slices

    def left_multiply(self, state):
        result = np.empty_like(state)
        for sl in self.ado_slices:
            ado = state[sl]
            result[sl] = self.lspace_op.left_multiply(ado)
        return result

    def right_multiply(self, state):
        result = np.empty_like(state)
        for sl in self.ado_slices:
            ado = state[sl]
            result[sl] = self.lspace_op.right_multiply(ado)
        return result

    def expectation_value(self, state):
        rho = state(self.ado_slices[0])
        return self.lspace_op.expectation_value(rho)

def matsubara_frequencies(K, gamma, T):
    """
    assuming gamma is the same for all sites
    """
    v = 2 * np.pi * T * np.arange(K + 1)
    v[0] = gamma
    return v

def corr_func_coeffs(K, gamma, T, reorg_en, matsu_freqs):
    """
    returns coefficients  c_{j,k} corresponding to the correlation function:
    C_j(t) = \sum_{m=0}^\inf c_{j,k} exp(-v_{j,k} t)


    doi: 10.1063/1.3271348
    """
    inv_T = 1 / T
    bath_coeffs = []

    bath_coeffs.append(reorg_en * gamma * (1 / np.tan(gamma / (2 * T)) - 1j))

    for k in xrange(1, K + 1):
        bath_coeffs.append(4 * reorg_en * gamma * T * matsu_freqs[k] /
                         (matsu_freqs[k] ** 2 - gamma ** 2))
    return bath_coeffs


def ADO_mappings(N, K, level_cutoff):
    """
    ADO (auxilary density operators) are indexed by a N by (K + 1) matrix
    consisting of non-negative integers.

    ADO_mappings calculates all possible matrices "ado_index" of size
    N by (K+1) where np.sum(m) < level_cutoff

    Parameters
    ----------
    N : integer
        number of states

    K : integer
        number of exponentials to include in the spectral density
        correlation function

    level_cutoff : integer
        number of levels at which to terminate the heiarchy expansion


    Returns
    -------
    ind_to_mat : list of matrices
                maps index to np.array
    mat_to_ind : function
                maps the np.array to the index

    ---------------------------------------------------------------------------
    Define S to be the set of all matrices of size N by (K + 1) with
    non-negative integer values.

    Define level L_i as:

    L_i = {m \in S | np.sum(m) == i}

    L_i can be found using the multichoose function. We will preserve the order
    that multichoose uses in ordering L_i

    L_i corresponds to the set of ADOs in the ith heiarchy.
    L_0 is a singleton set, corresponding to the RDO (reduced density matrix)
    """

    bins = N * (K + 1)

    permutations = []
    for c in range(level_cutoff):
        permutations.extend(multichoose(bins, c))

    inverted_permutations = {tuple(v): i for i, v in enumerate(permutations)}

    def mat_to_ind(mat):
        """maps np.array to index"""
        vec = mat.flatten()
        try:
            return inverted_permutations[tuple(vec)]
        except KeyError:
            return None

    ind_to_mat = [np.array(vec).reshape((N, K + 1)) for vec in permutations]
    return ind_to_mat, mat_to_ind


def multichoose(n, c):
    """
    stars and bars combinatorics problem:
    enumerates the different ways to parition c indistinguishable balls into
    n distinguishable bins.

    returns a list of n-length lists

    http://mathoverflow.net/a/9494
    """
    if c < 0 or n < 0:
        raise
    if not c:
        return [[0] * n]
    if not n:
        return []
    if n == 1:
        return [[c]]
    return [[0] + val for val in multichoose(n - 1, c)] + \
        [[val[0] + 1] + val[1:] for val in multichoose(n, c - 1)]


class HEOMModel(LiouvilleSpaceModel):

    """
    DynamicalModel for HEOM

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
        Hilbert subspace
    unit_convert : number, optional
        Unit conversion from energy to time units (default 1).

    References
    ----------
    """

    def __init__(self, hamiltonian, rw_freq=None, hilbert_subspace='gef',
                 unit_convert=1, level_cutoff=3, K=1, low_temp_corr=True):
        evolve_basis = 'site'
        super(HEOMModel, self).__init__(hamiltonian, rw_freq,
                                        hilbert_subspace, unit_convert,
                                        evolve_basis)
        self.level_cutoff = level_cutoff
        self.K = K
        self.low_temp_corr = low_temp_corr

    @memoized_property
    def evolution_super_operator(self):
        return (self.unit_convert
                * self.HEOM_tensor(self.hilbert_subspace,
                              K=self.K, level_cutoff=self.level_cutoff,
                              low_temp_corr=self.low_temp_corr))

    def equation_of_motion(self, liouville_subspace, heisenberg_picture=False):
        """
        Return the equation of motion for this dynamical model in the given
        subspace, a function which takes a state vector and returns its first
        time derivative, for use in a numerical integration routine
        """
        if heisenberg_picture:
            raise NotImplementedError('HEOM not implemented in the Heisenberg '
                                      'picture')

        # index = self.liouville_subspace_index(liouville_subspace)
        # mesh = np.ix_(index, index)
        # evolve_matrix = self.evolution_super_operator[mesh]
        evolve_matrix = self.evolution_super_operator
        size = evolve_matrix.shape
        hilbert_size_sq = self.hamiltonian.n_states(self.hilbert_subspace) ** 2

        # workaround: we save the pre/postprocessing functions as properties
        # this makes the HEOM a stateful object
        def _preprocess(rdo):
            ado_operator = np.zeros(size[0], dtype=np.complex128)
            ado_operator[:hilbert_size_sq] = rdo
            return ado_operator
        self.preprocess = _preprocess

        def postprocess(vec):
            return vec[:hilbert_size_sq]
        self.postprocess = postprocess

        def eom(t, rho_expanded):
            return evolve_matrix.dot(rho_expanded)
        return eom

    def HEOM_tensor(self, subspace='ge', K=3, level_cutoff=3, low_temp_corr=True):
        """
        Calculates the HEOM tensor elements in the energy eigenbasis

        the Liouville space is: (H^k) \tensordot (H^k)

        where H is the Hilbert space of the given subspace, and k is the total
        number of auxilary density matrices + 1 (the original density matrix)

        If the hilbert space is n x n, the total shape of the tensor is:

        [n*n*m,n*n*m]


            for n = 2, level_cutoff = 2

            p_00  p_01  p_10  p_11
        p_00[   ]|[   ]|[   ]|[   ]
            [   ]|[   ]|[   ]|[   ]
            -----|-----|-----|-----
        p_01[   ]|[   ]|[   ]|[   ]
            [   ]|[   ]|[   ]|[   ]
            -----|-----|-----|-----
        p_10[   ]|[   ]|[   ]|[   ]
            [   ]|[   ]|[   ]|[   ]
            -----|-----|-----|-----
        p_11[   ]|[   ]|[   ]|[   ]
            [   ]|[   ]|[   ]|[   ]

        Parameters
        ----------
        subspace : container, default 'ge'
            Container of any or all of 'g', 'e' and 'f' indicating the desired
            subspaces on which to calculate the HEOM tensor
        K : cutoff for the number of exponential functions to include in the
            correlation function
            (note: summation is 0-indexed, so K+1 exponentials are included)
        level_cutoff : level at which heiarchy is truncated

        Returns
        -------
        out : np.ndarray
            tensor giving the HEOM transfer rates between the
            density matrix elements and aux density matrix elements
            in the system energy eigenbasis

        References
        # 10.1021/jp109559p (indexing nomenclature refers to eqn. 9)
        doi: 10.1063/1.3271348 (eqn. A22)
        ----------
        """

        N = self.hamiltonian.n_states(subspace)
        gamma = self.hamiltonian.bath.cutoff_freq
        temp = self.hamiltonian.bath.temperature
        reorg_en = self.hamiltonian.bath.reorg_energy

        matsu_freqs = matsubara_frequencies(K, gamma, temp)
        bath_coeffs = corr_func_coeffs(K, gamma, temp, reorg_en, matsu_freqs)

        ado_indices, mat_to_ind = ADO_mappings(N, K, level_cutoff)

        tot_rho = len(ado_indices)
        L = lil_matrix((tot_rho * N * N, tot_rho * N * N), dtype=np.complex128)

        Isq = np.eye(N * N)
        self.ado_slices = [slice(n * N ** 2, (n + 1) * N ** 2) for n in
                                                range(len(ado_indices))]

        # unitary evolution:
        H = self.hamiltonian.H(subspace)
        liouvillian = super_commutator_sparse_matrix(H)

        # precompute the list of N vectorized projection operators
        proj_op_left = []
        proj_op_right = []

        for proj_op in self.hamiltonian.system_bath_couplings(
                       self.hilbert_subspace):
            proj_op_left.append(super_left_sparse_matrix(proj_op))
            proj_op_right.append(super_right_sparse_matrix(proj_op))

        matsu_freqs_inf = matsubara_frequencies(K + 5000, gamma, temp)
        bath_coeffs_inf = corr_func_coeffs(K + 5000, gamma, temp, reorg_en, matsu_freqs_inf)
        temp_corr_coeff = np.sum((bath_coeffs_inf / matsu_freqs_inf)[K + 1:])

        for n, ado_index in enumerate(ado_indices):
            # Loop over \dot{rho_n}
            left_slice = self.ado_slices[n]

            # diagonal shift:
            en_shift = np.sum(ado_index.dot(matsu_freqs))
            L[left_slice, left_slice] = -1j * liouvillian - Isq * en_shift

            #double commutator temperature correction!
            if temp_corr_coeff:
                temp = np.zeros((N ** 2, N ** 2))
                for j in xrange(N):
                    temp += (proj_op_left[j] + proj_op_right[j]
                            - 2 * proj_op_left[j].dot(proj_op_right[j]))
                L[left_slice, left_slice] -= temp_corr_coeff * temp

            print '\ncalculating ADO {}\n{}'.format(n, ado_index)
            # off-diagonal:
            for index, n_jk in np.ndenumerate(ado_index):
                # Loop over j and k (the sub-indices within the ado_index matrix)
                j, k = index
                ado_index[index] += 1
                p_index = mat_to_ind(ado_index)
                ado_index[index] -= 2
                n_index = mat_to_ind(ado_index)
                ado_index[index] += 1

                if p_index is not None:
                    plus_slice = self.ado_slices[p_index]
                    commutator = proj_op_left[j] - proj_op_right[j]
                    L[left_slice, plus_slice] = -1j * commutator

                if n_index is not None:
                    minus_slice = self.ado_slices[n_index]
                    commutator = bath_coeffs[k] * proj_op_left[j] \
                        - np.conjugate(bath_coeffs[k]) * proj_op_right[j]
                    L[left_slice, minus_slice] = -1j * n_jk * commutator
        return csr_matrix(L)
