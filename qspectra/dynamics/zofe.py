import numpy as np

from ..bath import PseudomodeBath
from ..utils import memoized_property

from .generic import DynamicalModel, SystemOperator

from .liouville_space import matrix_to_ket_vec


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
        self.operator = operator
        self.dynamical_model = dynamical_model

    def left_multiply(self, state):
        rho0, oop0 = self.dynamical_model.state_vec_to_operators(state)
        rho1 = self.operator.dot(rho0)
        #oop1 = np.einsum('cd,abde->abce', self.operator, oop0)
        oop1 = np.rollaxis(np.tensordot(self.operator, oop0, axes=[1, 2]), 0, 3) #faster than einsum
        return self.dynamical_model.operators_to_state_vec(rho1, oop1)

    def right_multiply(self, state):
        rho0, oop0 = self.dynamical_model.state_vec_to_operators(state)
        rho1 = rho0.dot(self.operator)
        oop1 = oop0.dot(self.operator)
        return self.dynamical_model.operators_to_state_vec(rho1, oop1)

    # @memoized_property
    # def expectation_value(self, state):
    #     rho0, _ = self.dynamical_model.state_vec_to_operators(state)
    #     # Proof: tr M rho = \sum_{ij} M_ij rho_ji
    #     return np.tensordot(self.operator, rho0, axes=([0, 1], [1, 0])) # faster than einsum
#-----------------------------------
    #NEW
    @memoized_property
    def expectation_value(self):
        def expec(state):
            rho0, _ = self.dynamical_model.state_vec_to_operators(state)
            # Proof: tr M rho = \sum_{ij} M_ij rho_ji
            return np.tensordot(self.operator, rho0, axes=([0, 1], [1, 0])) # faster than einsum
        return expec
#----------------------------------
    

class ZOFEModel(DynamicalModel):
    system_operator = ZOFESpaceOperator

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
        See references in method containing the ZOFE master equation
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

        self.oop_shape = (numb_pm, n_sit, n_stat, n_stat)

        

    def initial_state_and_oop_vec(self, initial_rho_vec):
        # initial auxiliary operator for the ZOFE master equation
        initial_oop = np.zeros(self.oop_shape, dtype=complex)
        return np.append(initial_rho_vec,
                         initial_oop.reshape((-1), order='F'))

    def ground_state(self, _):
        rho0 = self.hamiltonian.ground_state(self.hilbert_subspace)
        return self.initial_state_and_oop_vec(matrix_to_ket_vec(rho0))

    def map_between_subspaces(self, state, from_subspace, to_subspace):
        return state
        

    def state_vec_to_operators(self, rho_oop_vec):
        n_stat = self.hamiltonian.n_states(self.hilbert_subspace)
        n_stat_sq = n_stat**2
        rho = rho_oop_vec[:n_stat_sq].reshape((n_stat, n_stat), order='F')
        oop = rho_oop_vec[n_stat_sq:].reshape(self.oop_shape, order='F')
        return rho, oop

    def operators_to_state_vec(self, rho, oop):
        return np.append(rho.reshape((-1), order='F'),
                         oop.reshape((-1), order='F'))


    def rhodot_oopdot_vec(self, t, rho_oop_vec, oop_shape, ham, L_n, Gamma,
                          w, ham_hermit=False, rho_hermit=False):
        """
        Calculates the time derivatives rhodot and oopdot,
        i.e., of the density matrix and the auxiliary operator
        (takes and gives them back in vector form) according to the
        ZOFE master equation.
        Does work for one-exciton AND two-exciton space
        (including ground state).

        Arguments
        ---------
        t: time
        rho_oop_vec: vector containing the density matrix and the
            auxiliary operator at time t
        oop_shape: shape of the auxiliary operator, i.e., highest
            indices for each dimension.
            oop_shape should be (n_pseudomodes, n_sites, n_states, n_states)
        ham: Hamiltonian of the system part.
            ham is a 2D array of the form ham[state, state]
        L_n: system operator for the system-bath coupling.
            L_n is a 3D array of the form L_n[site, state, state]
        Gamma: =Omeg**2*huang, corresponding to a bath correlation
            spectrum with Lorentzians centered at frequencies Omeg with
            prefactors huang.
            Gamma is a 2D array of the form Gamma[pseudomode, site]
        w: =1j*Omeg+gamma, corresponding to a bath correlation spectrum
            with Lorentzians centered at frequencies Omeg with widths gammma.
            w is a 2D array of the form w[pseudomode, site]
        The following two parameters are needed to make the calculation more
        efficient if possible.
        ham_hermit: True if system Hamiltonian is hermitian
        rho_hermit: True if rho is hermitian

        Returns
        -------
        np.append(vec(rhodot), vec(oopdot)): time derivatives rhodot and oopdot
            in vector form.

        References
        ----------
        ZOFE master equation: Ritschel et. al., An efficient method to calculate
            excitation energy transfer in light-harvesting systems:
            application to the Fenna-Matthews-Olson complex, NJP 13 (2011) 113034
            (and references therein)
        Extend ZOFE master equation to two-exciton space: unpublished
        Speed up ZOFE master equation: unpublished

        """

        rho, oop = self.state_vec_to_operators(rho_oop_vec)

        sum_oop = oop.sum(axis=0) #sum over pseudomode index
    
        # ZOFE master equation

        a_op = np.tensordot(L_n.swapaxes(1,2).conj(), sum_oop, axes=([0, 2], [0, 1]))
        # tensordot is faster than einsum, so using tensordot when possible

        b_op = -1j*ham - a_op
        
        c_op = np.tensordot(np.tensordot(L_n, rho, axes=([2], [0])),
                            sum_oop.swapaxes(1,2).conj(), axes=([0, 2], [0, 1]))

        d_op = np.dot(b_op, rho) + c_op

        if not rho_hermit:
            big_operator = np.tensordot(np.tensordot(sum_oop, rho, axes=([2], [0])),
                                        L_n.swapaxes(1,2).conj(), axes=([0, 2], [0, 1]))
            if ham_hermit:
                f_op = rho.dot(b_op.T.conj()) + big_operator
            else:
                f_op = rho.dot(1j*ham - a_op.T.conj()) + big_operator
        else:
            if ham_hermit:
                f_op = d_op.T.conj()
            else:
                f_op = rho.dot(1j*ham - a_op.T.conj()) + c_op.T.conj()                                   
                                              
        rhodot = d_op + f_op

        # O operator evolution equation (uses b_op from above)
        oopdot = (np.einsum('ij,jkl->ijkl', Gamma, L_n)
                  - np.einsum('ij,ijkl->ijkl', w, oop)
                  + np.rollaxis(np.tensordot(b_op, oop, axes=[1, 2]), 0, 3)
                  - oop.dot(b_op))

        return self.operators_to_state_vec(rhodot, oopdot)


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
        # This is the function to build the coupling operators L_n that I had previously
        # used for ZOFE calculations and checked in the
        # cases of 'ge' and 'e' subspaces. However, in the two-exciton case 'gef', I have never checked
        # if ZOFE gives the right results.

        L_n = np.array([-1.*L for L in self.hamiltonian.system_bath_couplings(subspace=self.hilbert_subspace)])
        # NOTE THE MINUS SIGN!!
        # For the subspace cases 'ge' and 'e', the system_bath_couplings() method and the function
        # coupling_operator_L_n()
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

        sys_ham = self.hamiltonian.H(self.hilbert_subspace)

        def eom(t, rho_oop_vec):
            return (self.unit_convert
                    * self.rhodot_oopdot_vec(t, rho_oop_vec,
                                             self.oop_shape, sys_ham, L_n, Gamma, w))
        return eom
