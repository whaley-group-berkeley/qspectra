import numpy as np

from ..bath import PseudomodeBath
from ..utils import memoized_property

from .generic import DynamicalModel, SystemOperator
from .zofe_rhodot_oopdot import rhodot_oopdot_vec
from .vectorize_devectorize import vec


class ZOFESpaceOperator(SystemOperator):
    def __init__(self, operator, liouv_subspace_map, dynamical_model):
        """
        Parameters
        ----------
        operator : np.ndarray
            Matrix representation of the operator in the Hilbert subspace of
            `dynamical_model`.
        liouv_subspace_map : string
            String in the form 'eg->ee' indicating the mapping between
            Liouville subspaces on which the operator should act.
        dynamical_model : ZOFEModel
            ZOFE dynamical model on which this operator acts.
        """
        from_space, to_space = liouv_subspace_map.split('->')
        L1, R1 = map(dynamical_model.hilbert_subspace_indices, from_space)
        L2, R2 = map(dynamical_model.hilbert_subspace_indices, to_space)
        self.operator_L = operator[L2, L1]
        self.operator_R = operator[R1, R2]

    def left_multiply(self, state):
        rho0, oop0 = self.dynamical_model.state_vec_to_operators(state)
        rho1 = self.operator_L.dot(rho0)
        oop1 = np.einsum('cd,abde->abcd', self.operator_L, oop0)
        return self.operators_to_state_vec(rho1, oop1)

    def right_multiply(self, state):
        rho0, oop0 = self.dynamical_model.state_vec_to_operators(state)
        rho1 = rho0.dot(self.operator_R)
        oop1 = oop0.dot(self.operator_R)
        return self.operators_to_state_vec(rho1, oop1)

    @memoized_property
    def expectation_value(self, state):
        rho0, _ = self.dynamical_model.state_vec_to_operators(state)
        # Proof: tr M rho = \sum_{ij} M_ij rho_ji
        return np.einsum('ij,ji', self.operator_L, rho0)


class ZOFEModel(DynamicalModel):
    SystemOperator = ZOFESpaceOperator

    def __init__(self, hamiltonian, rw_freq=None, hilbert_subspace='gef',
                 unit_convert=1):
        """
        DynamicalModel for ZOFE master equation

        Assumes that each pigment is coupled to an identical, independent bath

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
        See references in module containing the ZOFE master equation
        """
        super(ZOFEModel, self).__init__(
             hamiltonian, rw_freq, hilbert_subspace, unit_convert)

        # initial auxiliary operator for the ZOFE master equation
        if not isinstance(self.hamiltonian.bath, PseudomodeBath):
            raise NotImplementedError('ZOFE only implemented for baths of type '
                                      'PseudomodeBath')

        numb_pm = self.hamiltonian.bath.numb_pm
        n_sit = self.hamiltonian.n_sites
        n_stat = self.hamiltonian.n_states(self.hilbert_subspace)

        self.initial_oop = np.zeros((numb_pm, n_sit, n_stat, n_stat), dtype=complex)
        self.initial_oop_vec = vec(self.initial_oop)


    def equation_of_motion(self, liouville_subspace, heisenberg_picture=False):
        """
        Return the equation of motion for this dynamical model in the given
        subspace, a function which takes a state vector and returns its first
        time derivative, for use in a numerical integration routine
        """

        if heisenberg_picture:
            raise NotImplementedError('ZOFE not implemented in the Heisenberg '
                                      'picture')

        #L_n = coupling_operator_L_n(n_sites, n_states, with_ground_state=False, two_excitons=False)
        # This is the function to build the coupling operators L_n that I had previously used for ZOFE calculations and checked in the
        # cases of 'ge' and 'e' subspaces. However, in the two-exciton case 'gef', I have never checked if ZOFE gives the
        # right results.

        L_n = [-1.*L for L in self.hamiltonian.system_bath_couplings(subspace=self.hilbert_subspace)]
        # NOTE THE MINUS SIGN!!
        # For the subspace cases 'ge' and 'e', the system_bath_couplings() method and the function coupling_operator_L_n()
        # that is commented out above give the same coupling operators.
        # BUT, for the two-exciton case 'gef', they give different results!
        # It is not clear yet, which of the two the right version for ZOFE is.
        # I have to go through my notes about the extension of ZOFE to the two-exciton space again.
        # So when using ZOFE for two-exciton calculations this issue has to be resolved first.

        # parameters of PseudomodeBath
        Omega = self.hamiltonian.bath.Omega
        gamma = self.hamiltonian.bath.gamma
        huang = self.hamiltonian.bath.huang

        Gamma = Omega**2*huang
        w = 1j*Omega+gamma

        oop_shape = self.initial_oop.shape

        sys_ham = self.hamiltonian.H(self.hilbert_subspace)

        def eom(t, rho_oop_vec):
            return self.unit_convert * rhodot_oopdot_vec(t, rho_oop_vec, oop_shape, sys_ham, L_n, Gamma, w)
        return eom
