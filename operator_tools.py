import numpy as np


def unit_vec(n, N, dtype=complex):
    """
    Returns the unit vector in direction n in N dimensions.
    """
    v = np.zeros(N, dtype=dtype)
    v[n] = 1
    return v


def basis_transform(X, U):
    """
    Transform X, a state vector, operator, vectorized operator or super-
    operator, into the basis given by the unitary transformation matrix U

    How to apply the transformation is inferred by the dimensions of X and U.

    Reference
    ---------
    Havel, T. F. Robust procedures for converting among Lindblad, Kraus and
    matrix representations of quantum dynamical semigroups. J Math. Phys.
    44, 534-557 (2003).
    """
    N = len(U)
    if X.shape == (N,):
        # X is a vector in Hilbert space
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
    raise ValueError('basis transformation incompatible with '
                     'operator dimensions')


def all_states(N, subspace='gef'):
    """
    List all states in the desired subspace for N pigments

    Assumes hard-core bosons (no double-excitations of the same state)

    Parameters
    ----------
    N : int
        Number of sites.
    subspace : container, default 'gef'
        Container of any or all of 'g', 'e' and 'f' indicating the desired
        subspaces on which the operator is defined.

    Returns
    -------
    states : list
        List of all states defined in the desired subspace, where each state is
        defined as a list of sites in the excited state
    """
    states = []
    if 'g' in subspace:
        states.append([])
    if 'e' in subspace:
        for i in xrange(N):
            states.append([i])
    if 'f' in subspace:
        for i in xrange(N):
            for j in xrange(i + 1, N):
                states.append([i, j])
    return states


def operator_1_to_2(operator1):
    """
    From the matrix representation of an operator in the 1-excitation subspace,
    determine its representation in the 2-excitation subspace

    Assumes that given the matrix element A_{nm}, the full representation of
    the operator is given by:
        \sum_{n,m} A_{nm} a^\dagger_n a_m

    Parameters
    ----------
    operator1 : np.ndarray
        Matrix representation of an operator defined on the 1-excitation
        subspace

    Returns
    -------
    operator2 : np.ndarray
        Matrix representation of the operator defined on the 2-excitation
        subspace
    """
    states = all_states(len(operator1), 'f')
    operator2 = np.zeros((len(states), len(states)), dtype=operator1.dtype)

    def delta(i, j):
        return int(i == j)

    for m in xrange(len(states)):
        for n in xrange(len(states)):
            (i, j), (k, l) = states[m], states[n]
            operator2[m, n] = (operator1[j, l] * delta(i, k) +
                               operator1[j, k] * delta(i, l) +
                               operator1[i, l] * delta(j, k) +
                               operator1[i, k] * delta(j, l))
    return operator2


def operator_extend(operator1, subspace='gef'):
    """
    Extend an operator defined in the 1-excitation subspace to include the
    ground and/or double-excitation subspaces

    Assumes that given the matrix element A_{nm}, the full representation of
    the operator is given by:
        \sum_{n,m} A_{nm} a^\dagger_n a_m

    Parameters
    ----------
    operator1 : np.ndarray
        Matrix representation of an operator defined on the 1-excitation
        subspace
    subspace : container, default 'gef'
        Container of any or all of 'g', 'e' and 'f' indicating the desired
        subspaces on which the operator is defined.

    Returns
    -------
    out : np.ndarray
        Matrix representation of the operator defined on the requested subspace
    """
    operators = []
    if 'g' in subspace:
        operators.append(np.array([[0]]))
    if 'e' in subspace:
        operators.append(operator1)
    if 'f' in subspace:
        operators.append(operator_1_to_2(operator1))

    sizes = [len(op) for op in operators]
    overall_size = sum(sizes)
    operator_extended = np.zeros((overall_size, overall_size),
                                 dtype=operator1.dtype)
    starts = np.cumsum([0] + sizes[:-1])
    ends = np.cumsum(sizes)
    for start, end, op in zip(starts, ends, operators):
        operator_extended[start:end, start:end] = op
    return operator_extended


def transition_operator(n, n_sites, subspace='gef', include_transitions='-+'):
    """
    Calculate the transition operator for creating an removing an excitation
    at site n of n_sites overall

    Parameters
    ----------
    n : int
        Site at which to alter the number of excitations (0-indexed).
    n_sites : int
        Number of sites.
    subspace : container, default 'gef'
        Container of any or all of 'g', 'e' and 'f' indicating the desired
        subspaces on which the operator is defined.
    include_transitions : str, default '-+'
        String containing '-' and/or '+' to indicating whether or not to
        annihilation and/or creation of an excitation.

    Returns
    -------
    out : np.ndarray
        Transition operator in matrix form
    """
    states = all_states(n_sites, subspace)
    dipole_matrix = np.zeros((len(states), len(states)))
    for i in xrange(len(states)):
        for j in xrange(len(states)):
            if (('+' in include_transitions and
                 states[i] == sorted(states[j] + [n]))
                or ('-' in include_transitions and
                    sorted(states[i] + [n]) == states[j])):
                dipole_matrix[i, j] = 1
    return dipole_matrix
