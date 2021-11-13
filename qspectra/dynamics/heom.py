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


def map_over_ados(func, state, result_size, ado_slices):
    result = np.zeros(result_size, dtype=complex)
    for to_slice, from_slice in ado_slices:
        ado = state[from_slice]
        result[to_slice] = func(ado)
    return result


class HEOMSpaceOperator(SystemOperator):
    def __init__(self, operator, liouv_subspace_map, dynamical_model):
        self.lspace_op = LiouvilleSpaceOperator(operator, liouv_subspace_map,
                                                dynamical_model.lspace_model)
        # self.ado_slices = dynamical_model.ado_slices
        self.ado_count = dynamical_model.ado_count

        to_size = self.lspace_op.to_indices.size
        from_size = self.lspace_op.from_indices.size
        self.ado_slices = [(slice(n * to_size, (n + 1) * to_size),
                            slice(n * from_size, (n + 1) * from_size))
                           for n in range(self.ado_count)]
        self.result_size = to_size * self.ado_count

    @property
    def bra_vector(self):
        lspace_bra = self.lspace_op.bra_vector
        bra = np.zeros(lspace_bra.size * self.ado_count, dtype=complex)
        bra[:lspace_bra.size] = lspace_bra
        return bra

    def left_multiply(self, state):
        return map_over_ados(self.lspace_op.left_multiply, state,
                             self.result_size, self.ado_slices)

    def right_multiply(self, state):
        return map_over_ados(self.lspace_op.right_multiply, state,
                             self.result_size, self.ado_slices)

    def commutator(self, state):
        return map_over_ados(self.lspace_op.commutator, state,
                             self.result_size, self.ado_slices)

    def expectation_value(self, state):
        _, from_slice = self.ado_slices[0]
        rho = state[from_slice]
        return self.lspace_op.expectation_value(rho)


def matsubara_frequencies(K, gamma, T):
    """
    assuming gamma is the same for all sites
    """
    v = 2 * np.pi * T * np.arange(K + 1)
    v[0] = gamma
    return v

def corr_func_coeffs(K, gamma, T, reorg_en, matsu_freqs, aki_temp_corr=False):
    """
    returns coefficients  c_{j,k} corresponding to the correlation function:
    C_j(t) = \sum_{m=0}^\inf c_{j,k} exp(-v_{j,k} t)


    doi: 10.1063/1.3271348
    """
    inv_T = 1 / T
    bath_coeffs = []

    if aki_temp_corr:
        # approx tan(x) = x
        bath_coeffs.append(reorg_en * gamma * (1/(gamma / (2 * T)) - 1j))
    else:
        bath_coeffs.append(reorg_en * gamma * (1 / np.tan(gamma / (2 * T)) - 1j))

    for k in range(1, K + 1):
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


class HEOMModel(DynamicalModel):

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
    system_operator = HEOMSpaceOperator

    def __init__(self, hamiltonian, rw_freq=None, hilbert_subspace='gef',
                 unit_convert=1, level_cutoff=3, K=1, low_temp_corr=True,
                 modified_HEOM=False, aki_temp_corr=False):
        evolve_basis = 'site'
        super(HEOMModel, self).__init__(hamiltonian, rw_freq,
                                        hilbert_subspace, unit_convert)

        self.lspace_model = LiouvilleSpaceModel(
            hamiltonian, rw_freq, hilbert_subspace, unit_convert, evolve_basis)
        self.level_cutoff = level_cutoff
        self.K = K
        self.low_temp_corr = low_temp_corr
        self.modified_HEOM = modified_HEOM
        self.aki_temp_corr = aki_temp_corr

        if modified_HEOM:
            # this low_temp_correction is a part of modified_HEOM
            assert self.low_temp_corr


        # calculate ADO properties
        N = self.hamiltonian.n_sites
        ado_indices, mat_to_ind = ADO_mappings(N, K, level_cutoff)
        self.ado_count = len(ado_indices)
        self.ado_indices = ado_indices
        self.mat_to_ind = mat_to_ind

    def equation_of_motion(self, liouville_subspace, heisenberg_picture=False):
        """
        Return the equation of motion for this dynamical model in the given
        subspace, a function which takes a state vector and returns its first
        time derivative, for use in a numerical integration routine
        """
        evolve_matrix = self.unit_convert * self.HEOM_tensor(liouville_subspace)

        if heisenberg_picture:
            evolve_matrix = evolve_matrix.T

        evolve_matrix = csr_matrix(evolve_matrix)

        def eom(t, rho_expanded):
            return evolve_matrix.dot(rho_expanded)

        return eom

    def thermal_state(self, liouville_subspace):
        rho0 = self.lspace_model.thermal_state(liouville_subspace)
        rho = np.zeros(rho0.size * self.ado_count, dtype=complex)
        rho[:rho0.size] = rho0
        return rho

    def dipole_operator(self, liouv_subspace_map, polarization,
                        transitions='-+'):
        """
        Return a dipole operator that follows the SystemOperator API for the
        given liouville_subspace_map, polarization and requested transitions.
        The operator will be defined in the same basis as self.evolve_basis
        """
        operator = self.hamiltonian.dipole_operator(self.hilbert_subspace,
                                                    polarization, transitions)
        return HEOMSpaceOperator(operator, liouv_subspace_map, self)

    def map_between_subspaces(self, state, from_subspace, to_subspace):

        def map_ss(liouville_space_state):
            return self.lspace_model.map_between_subspaces(
                liouville_space_state, from_subspace, to_subspace)

        from_size, to_size = [
            self.lspace_model.liouville_subspace_index(ss).size
            for ss in [from_subspace, to_subspace]]
        ado_slices = [(slice(n * to_size, (n + 1) * to_size),
                       slice(n * from_size, (n + 1) * from_size))
                      for n in range(self.ado_count)]
        result_size = to_size * self.ado_count
        return map_over_ados(map_ss, state, result_size, ado_slices)

    def density_matrix_to_state_vector(self, rho0, liouville_subspace):
        """
        turn a density matrix into a state vector to use as the
        diff eq initial condition
        """
        state0 = self.lspace_model.density_matrix_to_state_vector(rho0, liouville_subspace)
        HEOM_state0 = np.zeros(state0.size * self.ado_count, dtype=complex)
        HEOM_state0[:state0.size] = state0
        return HEOM_state0

    def state_vector_to_density_matrix(self, rhos):
        """
        turn the diff eq trajectory (list of state vectors) into a
        list of density matrices
        """
        Nsq = rhos.shape[-1] / self.ado_count
        rhos = rhos[:,:Nsq]
        temp = self.lspace_model.state_vector_to_density_matrix(rhos)
        return temp

    def HEOM_tensor(self, liouville_subspace):
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
        subspace = self.hilbert_subspace
        K = self.K
        level_cutoff = self.level_cutoff
        low_temp_corr = self.low_temp_corr

        ado_count = self.ado_count

        subspace_index = self.lspace_model.liouville_subspace_index(liouville_subspace)
        subspace_mesh = np.ix_(subspace_index, subspace_index)

        N = self.hamiltonian.n_states(subspace)
        M = subspace_index.size
        gamma = self.hamiltonian.bath.cutoff_freq
        temp = self.hamiltonian.bath.temperature
        reorg_en = self.hamiltonian.bath.reorg_energy

        matsu_freqs = matsubara_frequencies(K, gamma, temp)
        bath_coeffs = corr_func_coeffs(K, gamma, temp, reorg_en, matsu_freqs, self.aki_temp_corr)

        L = lil_matrix((ado_count * M, ado_count * M), dtype=np.complex128)

        Isq = np.eye(M)
        ado_slices = [slice(n * M, (n + 1) * M)
                      for n in range(ado_count)]

        # unitary evolution:
        H = self.hamiltonian.H(subspace)
        liouvillian = super_commutator_sparse_matrix(H)[subspace_mesh]

        # precompute the list of N vectorized projection operators
        proj_op_left = []
        proj_op_right = []

        for proj_op in self.hamiltonian.system_bath_couplings(
                       self.hilbert_subspace):
            proj_op_left.append(super_left_sparse_matrix(proj_op)[subspace_mesh])
            proj_op_right.append(super_right_sparse_matrix(proj_op)[subspace_mesh])

        matsu_freqs_inf = matsubara_frequencies(K + 5000, gamma, temp)
        bath_coeffs_inf = corr_func_coeffs(K + 5000, gamma, temp, reorg_en, matsu_freqs_inf, self.aki_temp_corr)

        if self.aki_temp_corr:
            temp_corr_coeff = bath_coeffs_inf[1] / matsu_freqs_inf[1]               # AKIs
        else:
            temp_corr_coeff = np.sum((bath_coeffs_inf / matsu_freqs_inf)[K + 1:])   # NOT AKIs

        for n, ado_index in enumerate(self.ado_indices):
            # Loop over \dot{rho_n}
            left_slice = ado_slices[n]

            # diagonal shift:
            en_shift = np.sum(ado_index.dot(matsu_freqs))
            L[left_slice, left_slice] = -1j * liouvillian - Isq * en_shift

            #double commutator temperature correction!
            if self.low_temp_corr or self.aki_temp_corr:
                temp = np.zeros((M, M))
                for proj_l, proj_r in zip(proj_op_left, proj_op_right):
                    temp += (proj_l + proj_r - 2 * proj_l.dot(proj_r))
                L[left_slice, left_slice] -= temp_corr_coeff * temp

            # off-diagonal:
            for index, n_jk in np.ndenumerate(ado_index):
                # Loop over j and k (the sub-indices within the ado_index matrix)
                j, k = index
                ado_index[index] += 1
                p_index = self.mat_to_ind(ado_index)
                ado_index[index] -= 2
                n_index = self.mat_to_ind(ado_index)
                ado_index[index] += 1

                if p_index is not None:
                    plus_slice = ado_slices[p_index]
                    commutator = proj_op_left[j] - proj_op_right[j]
                    if self.modified_HEOM:
                        mod_coef = np.sqrt((n_jk + 1) *
                                           np.abs(bath_coeffs[k]))
                    else:
                        mod_coef = 1
                    L[left_slice, plus_slice] = -1j * mod_coef * commutator

                if n_index is not None:
                    minus_slice = ado_slices[n_index]
                    commutator = bath_coeffs[k] * proj_op_left[j] \
                        - np.conjugate(bath_coeffs[k]) * proj_op_right[j]
                    if self.modified_HEOM:
                        mod_coef = np.sqrt(n_jk / np.abs(bath_coeffs[k]))
                    else:
                        mod_coef = n_jk

                    L[left_slice, minus_slice] = -1j * mod_coef * commutator
                    if self.aki_temp_corr:
                        commutator2 =  (proj_op_left[j] - proj_op_right[j])
                        L[left_slice, minus_slice] +=  (-1j * 4 * reorg_en * gamma ** 2 * temp / (matsu_freqs_inf[1] ** 2 - gamma ** 2)) * commutator2
        return L
