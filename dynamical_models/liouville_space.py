from collections import OrderedDict
import numpy as np

from .generic import DynamicalModel, SystemOperator
from ..utils import memoized_property


class SubspaceError(Exception):
    """
    Error class to indicate an invalid subspace
    """

def n_excitations(n_sites):
    return [1, n_sites, int(n_sites * (n_sites - 1) / 2)]


def n_states_from_n_sites(n_sites, subspace='gef'):
    """
    Returns the number of states from the number of sites
    """
    return sum(n_states for excitation, n_states
               in zip('gef', n_excitations(n_sites)) if excitation in subspace)


def n_sites_from_n_states(n_states, subspace='gef'):
    """
    Returns the number of sites from the number of states
    """
    g, e, f = [int(exc in subspace) for exc in 'gef']
    if f:
        # N = g + e * x + f * x * (x - 1) / 2
        return int((-2 * e + f
                    + np.sqrt((-2 * e + f) ** 2 - 8 * f * (g - n_states)))
                   / (2 * f))
    elif e:
        return n_states - g
    else:
        raise SubspaceError('cannot determine the number of site states from '
                            'only the number of ground states')


def extract_subspace(subspaces_string):
    """
    Given a string a subspace in Liouville space or a mapping between subspaces,
    returns the minimal containing Hilbert space subspace
    """
    return set(subspaces_string) - {',', '-', '>'}


def liouville_subspace_indices(liouville_subspace, subspace, n_sites):
    """
    Returns indices of the full vectorized density operator in the given
    subspace that are included in the indicated liouville subspace
    """
    total_states = 0
    cuts = {}
    for excitation, n_states in zip('gef', n_excitations(n_sites)):
        if excitation in subspace:
            cuts[excitation] = np.arange(n_states) + total_states
            total_states += n_states
    keep = np.zeros((total_states, total_states))
    for row, col in liouville_subspace.split(','):
        try:
            keep[np.ix_(cuts[row], cuts[col])] = 1
        except KeyError:
            raise SubspaceError("{}{} not in subspace '{}'".format(
                                row, col, subspace))
    return den_to_vec(keep).nonzero()[0]


def den_to_vec(rho_den):
    """
    Transform a density operator from matrix to vectorized (stacked column) form
    """
    return rho_den.reshape((-1), order='F')


def vec_to_den(rho_vec):
    """
    Transform a density operator from vectorized (stacked column) to matrix form
    """
    N = int(np.sqrt(np.prod(rho_vec.shape)))
    return rho_vec.reshape((N, N), order='F')


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


class LiouvilleSpaceOperator(SystemOperator):
    def __init__(self, operator, operator_subspace, liouv_subspace_map):
        """
        Parameters
        ----------
        operator : np.ndarray
            Matrix representation of the operator in the 0 to 2 excitation
            subspace.
        operator_subspace : container
            Container of any or all of 'g', 'e' and 'f' indicating the
            subspace on which the operator is defined.
        liouv_subspace_map : string
            String in the form 'eg,fe->gg,ee' indicating the mapping between
            Liouville subspaces on which the operator should act. Optionally,
            only one Liouville subspace may be provided (e.g., a string of the
            form 'eg,fe'), in which case the super operator is assumed to map
            from and to the same subspace.
        """
        self.operator = operator
        liouv_subspaces = (liouv_subspace_map.split('->')
                           if '->' in liouv_subspace_map
                           else [liouv_subspace_map, liouv_subspace_map])
        n_sites = n_sites_from_n_states(len(operator), operator_subspace)
        self.from_indices, self.to_indices = [
            liouville_subspace_indices(liouv_subspace, operator_subspace, n_sites)
            for liouv_subspace in liouv_subspaces]
        self.super_op_mesh = np.ix_(self.to_indices, self.from_indices)

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
