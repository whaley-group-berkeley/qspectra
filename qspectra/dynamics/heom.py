import numpy as np
# import scipy as sp
from scipy import constants
from scipy.sparse import lil_matrix, csr_matrix

# from ..bath import PseudomodeBath
# from .base import DynamicalModel, SystemOperator
# from .liouville_space import matrix_to_ket_vec

# from ..operator_tools import basis_transform_operator

from .liouville_space import (LiouvilleSpaceModel, super_commutator_matrix,
                              super_left_matrix, super_right_matrix)
from ..utils import memoized_property

K_CM = constants.physical_constants[
    'Boltzmann constant in inverse meters per kelvin'][0] / 100

def matsubara_frequencies(M, gamma, T):
    """
    assuming gamma is the same for all sites
    """
    coef = 2 * np.pi * K_CM * T
    v = np.arange(M + 1) * coef
    v[0] = gamma
    return v

def corr_func_coeffs(M, gamma, T, reorg_en, matsu_freqs):
    """
    returns coefficients  c_{j,m} corresponding to the correlation function:
    C_j(t) = \sum_{m=0}^\inf c_{j,m} exp(-v_{j,m} t)


    doi: 10.1063/1.3271348
    """
    inv_T = 1 / (K_CM * T)
    bath_coeffs = []

    bath_coeffs.append(reorg_en * gamma * (1 / np.tan(inv_T * gamma / 2) - 1j))

    for m in xrange(1, M + 1):
        bath_coeffs.append(4 * reorg_en * gamma * matsu_freqs[m] / (inv_T *
                         (matsu_freqs[m] ** 2 - gamma ** 2)))
    return bath_coeffs

def HEOM_tensor(hamiltonian, subspace='ge', M=3, level_cutoff=3):
    """
    Calculates the HEOM tensor elements in the energy eigenbasis

    the Liouville space is: (H^k) \tensordot (H^k)

    where H is the Hilbert space of the given subspace, and k is the total
    number of auxilary density matrices + 1 (the original density matrix)

    If the hilbert space is n x n, the total shape of the tensor is:

    [n*n*m,n*n*m]


        for n = 2, level_cutoff = 3

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
    hamiltonian : hamiltonian.Hamiltonian
        Hamiltonian object specifying the system
    subspace : container, default 'ge'
        Container of any or all of 'g', 'e' and 'f' indicating the desired
        subspaces on which to calculate the HEOM tensor
    M : cutoff for the number of exponential functions to include in the
        correlation function
        (note: summation is 0-indexed, so M+1 exponentials are included)
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

    N = hamiltonian.n_states(subspace)
    gamma = hamiltonian.bath.cutoff_freq
    temp = hamiltonian.bath.temperature
    reorg_en = hamiltonian.bath.reorg_energy

    matsu_freqs = matsubara_frequencies(M, gamma, temp)
    bath_coeffs = corr_func_coeffs(M, gamma, temp, reorg_en, matsu_freqs)

    rho_indices, mat_to_ind = ADO_mappings(N, M, level_cutoff)
    tot_rho = len(rho_indices)
    print 'there are {} ADOs in total'.format(tot_rho)

    L = lil_matrix((tot_rho * N * N, tot_rho * N * N), dtype=np.complex128)

    # Tensor variables:
    Isq = np.eye(N * N)
    Nsq = N ** 2

    # unitary evolution:
    H = np.diag(hamiltonian.E(subspace))
    liouvillian = -1j * super_commutator_matrix(H)

    # precompute the list of N vectorized projection operators
    proj_op_left = []
    proj_op_right = []
    print 'creating projection operators'
    for n in xrange(N):
        proj_op = np.zeros((N, N))
        proj_op[n, n] = 1
        proj_op_left.append(super_left_matrix(proj_op))
        proj_op_right.append(super_right_matrix(proj_op))

    print 'creating master equation'
    for n, rho_index in enumerate(rho_indices):
        # Loop over \dot{rho_n}
        left_slice = slice(n * Nsq, (n + 1) * Nsq)

        # diagonal shift:
        en_shift = -np.sum(rho_index.dot(matsu_freqs))
        L[left_slice, left_slice] = liouvillian + Isq * en_shift

        #double commutator temperature correction!

        print '\ncalculating ADO {}\n{}'.format(n, rho_index)
        # off-diagonal:
        for index, n_jk in np.ndenumerate(rho_index):
            # Loop over j and k (the sub-indices within the rho_index matrix)
            j, k = index
            rho_index[index] += 1
            p_index = mat_to_ind(rho_index)
            rho_index[index] -= 2
            n_index = mat_to_ind(rho_index)
            rho_index[index] += 1

            if p_index is not None:
                plus_slice = slice(p_index * Nsq, (p_index + 1) * Nsq)
                commutator = proj_op_left[j] - proj_op_right[j]
                L[left_slice, plus_slice] = -1j * commutator
                # L[plus_slice, left_slice] = -1j * commutator

            if n_index is not None:
                minus_slice = slice(n_index * Nsq, (n_index + 1) * Nsq)
                commutator = bath_coeffs[k] * proj_op_left[j] \
                    - np.conjugate(bath_coeffs[k]) * proj_op_right[j]
                L[left_slice, minus_slice] = -1j * n_jk * commutator
                # L[minus_slice, left_slice] = -1j * n_jk * commutator
    print L.shape
    return csr_matrix(L)


def ADO_mappings(N, K, level_cutoff):
    """
    --------------------
    Define V to be the set of all vectors of length N*(K+1) with non-negative
    integer values.

    The level P_c is defined:

    P_c = {v \in V | sum(v) == c}

    The total set we want is:

    P = {v \in V | sum(v) <= level_cutoff} = {v \in P_c | c <= level_cutoff}

    P_c can be found using the multichoose function. We will preserve the order
    that multichoose uses in ordering P
    --------------------
    the elements in P enumerate density matrices (either the RDM or ADMs)
    P_0 is the singleton set that maps to the RDM
    P_c is set of indices that maps to the ADMs at the hierarchy level c
    --------------------
    Returns two functions:
    ind_to_mat maps index to np.array
    mat_to_ind maps the np.array to the index
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
                 unit_convert=1, level_cutoff=3, M=1):
        evolve_basis = 'eigen'
        super(HEOMModel, self).__init__(hamiltonian, rw_freq,
                                        hilbert_subspace, unit_convert,
                                        evolve_basis)
        self.level_cutoff = level_cutoff
        self.M = M

    @memoized_property
    def evolution_super_operator(self):
        return (self.unit_convert
                * HEOM_tensor(self.hamiltonian, self.hilbert_subspace,
                              M=self.M, level_cutoff=self.level_cutoff))

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

