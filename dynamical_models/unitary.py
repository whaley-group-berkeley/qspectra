import numpy as np

from .liouville_space import liouville_subspace_indices, LiouvilleSpaceModel
from ..utils import memoized_property


class UnitaryModel(LiouvilleSpaceModel):
    def __init__(self, hamiltonian, rw_freq=None, hilbert_subspace='gef',
                 unit_convert=1):
        """
        Dynamical model for unitary evolution
        
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
        super(UnitaryModel, self).__init__(hamiltonian, rw_freq,
                                           hilbert_subspace)
        self.unit_convert = unit_convert
        
    @memoized_property
    def unitary_super_operator(self):
        return self.unit_convert*self.hamiltonian.H(self.hilbert_subspace)
        
    def equation_of_motion(self, liouville_subspace):
        """
        Return the equation of motion for this dynamical model in the given
        subspace, a function which takes a state vector and returns its first
        time derivative, for use in a numerical integration routine
        """
        index = liouville_subspace_indices(
                                        liouville_subspace,
                                        self.hilbert_subspace,
                                        self.hamiltonian.n_sites,
                                        self.hamiltonian.n_vibrational_states)
        mesh = np.ix_(index, index)
        evolve_matrix = self.unitary_super_operator[mesh]
        def eom(t, rho):
            return evolve_matrix.dot(rho)
        return eom
        
    @property
    def time_step(self):
        """
        The default time step at which to sample the equation of motion (in the
        rotating frame)
        """
        return self.hamiltonian.time_step / self.unit_convert
