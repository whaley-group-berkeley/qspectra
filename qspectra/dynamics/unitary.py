from .liouville_space import LiouvilleSpaceModel, super_commutator_matrix
from ..utils import memoized_property


class UnitaryModel(LiouvilleSpaceModel):
    @memoized_property
    def evolution_super_operator(self):
        H = self.unit_convert * self.hamiltonian.H(self.hilbert_subspace)
        return -1j * super_commutator_matrix(H)
