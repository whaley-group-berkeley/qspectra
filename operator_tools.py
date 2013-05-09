from functools import wraps
import itertools
import numpy as np
import scipy.sparse

from spectra.utils import MetaArray, memoize


class BasisError(Exception):
    """Exception to indicate an unsupported choice of basis"""


class SubspaceError(Exception):
    """Exception to indicate an invalid subspace"""


def den_to_vec(rho_den):
    """Transform a density operator from matrix to vectorized (stacked column)
    form"""
    return rho_den.reshape((-1), order='F')


def vec_to_den(rho_vec):
    """Transform a density operator from vectorized (stacked column) to matrix
    form"""
    N = int(np.sqrt(np.prod(rho_vec.shape)))
    return rho_vec.reshape((N, N), order='F')


def vec_pops(rho_vec):
    """Extract the population terms from a vectorized density matrix"""
    return np.diag(vec_to_den(rho_vec)).real


def exclude_ground(rho_vec):
    """Exclude ground station population and coherence terms from the density
    matrix"""
    return den_to_vec(vec_to_den(rho_vec)[1:, 1:])


def meta_series_op(func):
    """Call on an operator that acts on states to turn it into an operator
    that acts on a series of states contained in a MetaArray object"""
    @wraps(func)
    def wrapper(series):
        return MetaArray(map(func, series), **series.metadata)
    return wrapper


def normalized_exclude_ground(vec_series):
    """Returns the time-series of excited state populations normalized by the
    final total excited state population"""
    return (meta_series_op(exclude_ground)(vec_series)
            / (1 - vec_series[-1, 0]))


def normalized_pops(rho_vec_series):
    """Returns the time-series of excited state populations normalized by the
    final total excited state population"""
    pops = meta_series_op(vec_pops)(rho_vec_series)
    return pops[:, 1:] / (1 - pops[-1, 0])


def unit_vec(n, N):
    v = np.zeros(N, dtype=complex)
    v[n] = 1
    return v


def diag_vec(N):
    return den_to_vec(np.diag(np.ones(N, dtype=complex)))


def S_commutator(M):
    I = np.identity(M.shape[0])
    return np.kron(I, M) - np.kron(M.T, I)


def S_left(M):
    I = np.identity(M.shape[0])
    return np.kron(I, M)


def S_right(M):
    I = np.identity(M.shape[0])
    return np.kron(M.T, I)


def tensor_to_super(R_tensor):
    N = R_tensor.shape[0]
    R_matrix = np.empty((N ** 2, N ** 2), dtype=complex)
    for i in xrange(N):
        for j in xrange(N):
            R_matrix[i::N, j::N] = R_tensor[i, :, j, :]
    return R_matrix


@memoize
def density_subset(parts, N_1):
    """Returns indices of the minimal vectorized density matrix elements in
    specified parts."""
    N_2 = 1 + N_1 + N_1 * (N_1 - 1) / 2
    cuts = {'g': np.array([0], dtype=int),
            'e': np.arange(1, N_1 + 1, dtype=int),
            'f': np.arange(N_1 + 1, N_2, dtype=int)}
    keep = np.zeros((N_2, N_2), dtype=int)
    if parts is None:
        keep[:] = 1
    else:
        for row, col in parts.split(','):
            keep[np.ix_(cuts[row], cuts[col])] = 1
    return den_to_vec(keep).nonzero()[0]


def density_pop_indices(N):
    rho = np.identity(N)
    return np.where(den_to_vec(rho) > 0)[0]


@memoize
def all_states(N, subspace='gef'):
    """List all states in the 0-2 excitation subspace for N pigments"""
    states = []
    if 'g' in subspace or '0' in subspace:
        states.append([])
    if 'e' in subspace or '1' in subspace:
        for i in xrange(N):
            states.append([i])
    if 'f' in subspace or '2' in subspace:
        for i in xrange(N):
            for j in xrange(i + 1, N):
                states.append([i, j])
    return states


def operator_1_to_2(operator1, K_2=None):
    """Convert single-excitation subspace operator to the double-excitation
    subspace"""
    states = all_states(len(operator1), 'f')
    operator2 = np.zeros((len(states), len(states)), dtype=operator1.dtype)

    def delta(i, j):
        return int(i == j)

    for (m, n) in itertools.product(xrange(len(states)), repeat=2):
        (i, j), (k, l) = states[m], states[n]
        operator2[m, n] = (operator1[j, l] * delta(i, k) +
                           operator1[j, k] * delta(i, l) +
                           operator1[i, l] * delta(j, k) +
                           operator1[i, k] * delta(j, l))
        if K_2 is not None and m == n:
            operator2[m, m] += K_2[m]

    return operator2


def operator_extend(operator1, subspace='gef', K_2=None):
    operators = []
    for space in subspace:
        if space == '0' or space == 'g':
            operators.append(np.array([[0]]))
        elif space == '1' or space == 'e':
            operators.append(operator1)
        elif space == '2' or space == 'f':
            operators.append(operator_1_to_2(operator1, K_2))
        else:
            raise SubspaceError('Invalid subspace {0}'.format(space))
    sizes = [len(op) for op in operators]
    overall_size = sum(sizes)
    operator_extended = np.zeros((overall_size, overall_size),
                                 dtype=operator1.dtype)
    starts = np.cumsum([0] + sizes[:-1])
    ends = np.cumsum(sizes)
    for op, (start, end) in zip(operators, zip(starts, ends)):
        operator_extended[start:end, start:end] = op
    return operator_extended


@memoize
def transition_operator(n, N, subspace='gef'):
    """Calculate the transition operator for creating an removing an excitation
    at site n of N sites overall"""
    states = all_states(N, subspace)
    dipole_matrix = np.zeros((len(states), len(states)))
    for (i, j) in itertools.product(range(len(states)), repeat=2):
        if ((sorted(states[i] + [n]) == sorted(states[j])) or
                sorted(states[i]) == sorted(states[j] + [n])):
            dipole_matrix[i, j] = 1
    return dipole_matrix


def transform_out_sparse(default=False, sparsifier=scipy.sparse.csr_matrix):
    """Function that returns a decorator to add optional basis or sparifying
    transformations to a function that returns states or operators
    """
    def decorator(func):
        @wraps(func)
        def wrapper(hamiltonian, *args, **kwargs):
            sparse = kwargs.pop('sparse', default)
            X = func(hamiltonian, *args, **kwargs)
            if sparse:
                X = sparsifier(X)
            return X
        return wrapper
    return decorator


def to_basis(U, X):
    """Return operator X in basis given by the unitary transform U"""
    N = len(U)
    if X.shape == (N,):
        # X is a vector
        return U.T.conj().dot(X)
    elif X.shape == (N, N):
        # X is an operator
        return U.T.conj().dot(X).dot(U)
    elif X.shape[0] == N ** 2:
        # X is in Liouville space
        U_S = np.kron(U, U)
        if X.shape == (N ** 2,):
            # X is a vectorized operator
            return U_S.T.conj().dot(X)
        elif X.shape == (N ** 2, N ** 2):
            # X is a super-operator
            return U_S.T.conj().dot(X).dot(U_S)
    raise BasisError('basis transformation incompatible with '
                     'operator dimensions')
