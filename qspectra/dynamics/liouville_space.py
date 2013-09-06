from abc import abstractproperty
import itertools
import numpy as np

from .generic import DynamicalModel, SystemOperator
from ..operator_tools import SubspaceError, n_excitations
from ..utils import memoized_property


def liouville_subspace_indices(liouville_subspace, full_subspace, n_sites,
                               n_vibrational_states=1):
    """
    Returns indices of the full vectorized density operator in the given
    subspace that are included in the indicated liouville subspace
    """
    total_states = 0
    cuts = {}
    exc_n_states = n_excitations(n_sites, n_vibrational_states)
    for excitation, n_states in zip('gef', exc_n_states):
        if excitation in full_subspace:
            cuts[excitation] = np.arange(n_states) + total_states
            total_states += n_states
    keep = np.zeros((total_states, total_states))
    for row, col in liouville_subspace.split(','):
        try:
            keep[np.ix_(cuts[row], cuts[col])] = 1
        except KeyError:
            raise SubspaceError("{}{} not in subspace '{}'".format(
                                row, col, full_subspace))
    return matrix_to_ket_vec(keep).nonzero()[0]


def all_liouville_subspaces(hilbert_subspace):
    """
    Given a Hilbert subspace, return a comma separated list with all included
    Liouville subspaces

    Example
    -------
    >>> all_liouville_subspaces('ge')
    'gg,ge,eg,ee'
    """
    return ','.join(''.join(s) for s
                    in itertools.product(hilbert_subspace, repeat=2))


def matrix_to_ket_vec(matrix):
    """
    Transform an operator from matrix to vectorized (stacked column) form
    """
    return matrix.reshape((-1), order='F')


def ket_vec_to_matrix(ket_vec):
    """
    Transform an operator vectorized (stacked column) form into a matrix
    """
    N = int(np.sqrt(np.prod(ket_vec.shape)))
    return ket_vec.reshape((N, N), order='F')


def matrix_to_bra_vec(matrix):
    """
    Transform an operator from matrix to vectorized (stacked row) form
    """
    return matrix.reshape((-1), order='C')


def tensor_to_super(tensor_operator):
    """
    Transform a linear operator on density matrices from tensor to super-
    operator form

    Parameters
    ----------
    tensor_operator : np.ndarray
        Four dimensional tensor describing the linear operator on a density
        matrix.

    Returns
    -------
    super_operator : np.ndarray
        Two dimensional super-operator giving the equivalent linear operator
        that acts on liouville state vectors.
    """
    N = tensor_operator.shape[0]
    super_operator = np.empty((N ** 2, N ** 2), dtype=tensor_operator.dtype)
    for i in xrange(N):
        for j in xrange(N):
            super_operator[i::N, j::N] = tensor_operator[i, :, j, :]
    return super_operator


def super_commutator_matrix(operator):
    """
    Returns the super-operator that when applied to a vectorized density
    operator is equivalent to the commutator of the operator and the density
    matrix
    """
    return super_left_matrix(operator) - super_right_matrix(operator)


def super_left_matrix(operator):
    """
    Returns the super-operator that when applied to a vectorized density
    operator is equivalent to the matrix product of the operator times the
    density matrix

    Reference
    ---------
    Havel, T. F. Robust procedures for converting among Lindblad, Kraus and
    matrix representations of quantum dynamical semigroups. J Math. Phys.
    44, 534-557 (2003).
    """
    I = np.identity(len(operator))
    return np.kron(I, operator)


def super_right_matrix(operator):
    """
    Returns the super-operator that when applied to a vectorized density
    operator is equivalent to the matrix product of the density matrix times the
    operator

    Reference
    ---------
    Havel, T. F. Robust procedures for converting among Lindblad, Kraus and
    matrix representations of quantum dynamical semigroups. J Math. Phys.
    44, 534-557 (2003).
    """
    I = np.identity(len(operator))
    return np.kron(operator.T, I)


class LiouvilleSpaceModel(DynamicalModel):
    def __init__(self, hamiltonian, rw_freq=None, hilbert_subspace='gef',
                 unit_convert=1):
        """
        DynamicalModel for Liouville space

        The equation_of_motion method needs to be defined by a subclass

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

        References
        ----------
        Nitzan (2006)
        """
        self.hamiltonian = hamiltonian.in_rotating_frame(rw_freq)
        self.rw_freq = self.hamiltonian.energy_offset
        self.hilbert_subspace = hilbert_subspace
        self.unit_convert = unit_convert

    def _liouville_subspace_indices(self, liouville_subspace, subspace=None):
        if subspace is None:
            subspace = self.hilbert_subspace
        return liouville_subspace_indices(liouville_subspace, subspace,
                                          self.hamiltonian.n_sites,
                                          self.hamiltonian.n_vibrational_states)

    def ground_state(self, liouville_subspace):
        rho0 = self.hamiltonian.ground_state(self.hilbert_subspace)
        index = self._liouville_subspace_indices(liouville_subspace)
        return matrix_to_ket_vec(rho0)[index]

    @abstractproperty
    def evolution_super_operator(self):
        pass

    def equation_of_motion(self, liouville_subspace, heisenberg_picture=False):
        """
        Return the equation of motion for this dynamical model in the given
        subspace, a function which takes a state vector and returns its first
        time derivative, for use in a numerical integration routine
        """
        index = self._liouville_subspace_indices(liouville_subspace)
        mesh = np.ix_(index, index)
        evolve_matrix = self.evolution_super_operator[mesh]
        if heisenberg_picture:
            # This works because these two equations of motion are equivalent:
            #     rho.reshape(-1, 1).dot(L)
            # and:
            #     L.T.dot(rho)
            evolve_matrix = evolve_matrix.T
        def eom(t, rho):
            return evolve_matrix.dot(rho)
        return eom

    def map_between_subspaces(self, state, from_subspace, to_subspace):
        from_indices, to_indices = map(self._liouville_subspace_indices,
                                       [from_subspace, to_subspace])
        N = self.hamiltonian.n_states(self.hilbert_subspace)
        new_state = matrix_to_ket_vec(np.zeros((N, N), dtype=complex))
        new_state[from_indices] = state
        return new_state[to_indices]

    @property
    def time_step(self):
        """
        The default time step at which to sample the equation of motion (in the
        rotating frame)
        """
        return self.hamiltonian.time_step / self.unit_convert

    def dipole_operator(self, liouv_subspace_map, polarization,
                        transitions='-+'):
        operator = self.hamiltonian.dipole_operator(self.hilbert_subspace,
                                                    polarization, transitions)
        return LiouvilleSpaceOperator(operator, liouv_subspace_map, self)


class LiouvilleSpaceOperator(SystemOperator):
    def __init__(self, operator, liouv_subspace_map, dynamical_model):
        """
        Parameters
        ----------
        operator : np.ndarray
            Matrix representation of the operator in the Hilbert subspace of
            `dynamical_model`.
        liouv_subspace_map : string
            String in the form 'eg,fe->gg,ee' indicating the mapping between
            Liouville subspaces on which the operator should act. Optionally,
            only one Liouville subspace may be provided (e.g., a string of the
            form 'eg,fe'), in which case the super operator is assumed to map
            from and to the same subspace.
        dynamical_model : LiouvilleSpaceModel
            LiouvilleSpaceModel on which this operator acts.
        """
        self.operator = operator
        liouv_subspaces = (liouv_subspace_map.split('->')
                           if '->' in liouv_subspace_map
                           else [liouv_subspace_map, liouv_subspace_map])
        self.from_indices, self.to_indices = \
            map(dynamical_model._liouville_subspace_indices, liouv_subspaces)
        self.super_op_mesh = np.ix_(self.to_indices, self.from_indices)

    @property
    def bra_vector(self):
        # cast the operator to a complex vector so integrate methods handle it
        # properly
        operator = np.asanyarray(self.operator, dtype=complex)
        return matrix_to_bra_vec(operator)[self.from_indices]

    @memoized_property
    def _super_left_matrix(self):
        # save the matrix for left multiplication so it can be used by both the
        # left_multply and expectation_value methods
        return super_left_matrix(self.operator)[self.super_op_mesh]

    @property
    def left_multiply(self):
        return self._super_left_matrix.dot

    @memoized_property
    def right_multiply(self):
        return super_right_matrix(self.operator)[self.super_op_mesh].dot

    @memoized_property
    def commutator(self):
        return super_commutator_matrix(self.operator)[self.super_op_mesh].dot

    @memoized_property
    def expectation_value(self):
        # Derivation:
        # tr M rho = tr super_left(M) vec(rho)
        #          = tr_M vec(rho)
        #          = sum_{ij} vec(I)_i S_ij v_j
        # => (tr_M)_j = sum_i vec(I)_i S_ij
        tr = np.identity(len(self.operator)).reshape(-1)[self.to_indices]
        return tr.dot(self._super_left_matrix).dot
