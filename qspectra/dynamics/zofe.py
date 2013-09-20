import numpy as np

from .liouville_space import LiouvilleSpaceModel
from ..utils import memoized_property
from .zofe_rhodot_oopdot import rhodot_oopdot_vec
from .vectorize_devectorize import vec
from ..bath import PseudomodeBath

# def ee_index_map(N):
#     """
#     Builds matrix M so that M[n,m]=i, where n and m are the labels of the two excited monomers
#     in the double-exciton states and i is the basis index of the corresponding double-exciton state.
#     Needs: 
#     import numpy as np
#     JR20111111
#     """
#     M = np.zeros((N,N),int)
#     i = N
#     for n in range(N):
#         for m in range(n + 1, N):
#             i += 1
#             M[n,m] = i
#     return M


# def coupling_operator_L_n(numb_monomers, BS, with_ground_state=False, two_excitons=False):
#     """
#     Builds the system operator L_n for the system-bath coupling that is used
#     in the ZOFE master equation.
#     Needs: 
#     import numpy as np
#     ee_index_map
#     JR20130227.
#     """
#     # If Stephan's system_bath_couplings(subspace) method of the hamiltonia class does the same as this function, we can drop this function. This has to be checked.
    
#     ee_ind = ee_index_map(numb_monomers)
#     L_n = np.zeros((numb_monomers,BS,BS), dtype=complex)
#     for n in np.arange(numb_monomers):
#         if not with_ground_state:
#             L_n[n,n,n] -= 1.  # single exciton part WITHOUT ground state, i.e. numb_monomers = BS
#         if with_ground_state:
#             L_n[n,n+1,n+1] -= 1.  # single exciton part WITH ground state, i.e. numb_monomers = BS-1
#             if two_excitons:
#                 for m in np.arange(numb_monomers):
#                     i = ee_ind[n,m]
#                     if not i==0: L_n[n,i,i] -= 2. # two exciton part WITH ground state
#     return L_n


class ZOFEModel(LiouvilleSpaceModel):
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
        super(ZOFEModel, self).__init__(hamiltonian, rw_freq,
                                            hilbert_subspace, unit_convert)

        # initial auxiliary operator for the ZOFE master equation
        if not isinstance(self.hamiltonian.bath, PseudomodeBath):
            raise NotImplementedError

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
            raise NotImplementedError

        
                
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

    # this is just a dummy. Should probably do this more elegantly somehow, without this dummy.
    @memoized_property
    def evolution_super_operator(self):
        return "nothing"

    
