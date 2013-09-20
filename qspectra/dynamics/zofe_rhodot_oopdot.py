#started: 26.10.2011
#JR
#last modified: 28.02.2013

#
# Checks for one-exciton space version:
#
# 1) Results agreed perfectly with analytic linear NMQSD Monomer absorption spectrum  (JR20111026).
# 2) Results agreed perfectly with numeric linear NMQSD ZOFE dimer absorption spectrum calculated with "NMQSD_ZOFE_dimer_zero_temp_absorp_spec_numeric.py" (JR20111026).
# 3) Checked energy transfer qualitatively --> found sign error in equation for rhodot (JR20111029).
#
# 4) Checked against one-exciton version with ODE optimized for one-exciton coupling operator L_n: results agreed perfectly, but optimized version is significantly faster (18 seconds versus 28 seconds) (JR20111112).
#
# 5) New optimized efficient version checked against old inefficient version.
#    Results agreed perfectly (at least up to 7 decimals) for trimer with 2 pseudomodes
#    and new version was faster by more than factor 3
#    (expect that this speed up will strongly increase with the number of monomers and
#    the number of pseudomodes) JR20130228.

import numpy as np

#from ..operator_tools import dag
from .vectorize_devectorize import (vec, mat, tens)

def dag(mat):
    """Returns conjugate transpose of matrix."""
    # we should put this function somewhere else, not sure where is best.
    # Maybe to operator_tool.py
    return mat.transpose().conj()


# function containing the ODE  =========================
def rhodot_oopdot_vec(t, rho_oop_vec, oop_shape, ham, L_n, Gamma, w):

    """
    Calculates the time derivatives rhodot and oopdot,
    i.e., of the density matrix and the auxiliary operator
    (takes and gives them back in vector form) according to the
    ZOFE master equation.
    Does work for one-exciton AND two-exciton space (including ground state).

    Arguments
    ---------
    t: time
    rho_oop_vec: vector containing the density matrix and the auxiliary operator at time t
    oop_shape: shape of the auxiliary operator, i.e., highest indices for each dimension
    ham: Hamiltonian of the system part
    L_n: system operator for the system-bath coupling
    Gamma: =Omeg**2*huang, corresponding to a bath correlation spectrum with Lorentzians centered at frequencies Omeg with prefactors huang
    w: =1j*Omeg+gamma, corresponding to a bath correlation spectrum with Lorentzians centered at frequencies Omeg with widths gammma

    Returns
    -------
    np.append(vec(rhodot), vec(oopdot)): time derivatives rhodot and oopdot in vector form.

    References
    ----------
    ZOFE master equation: Ritschel et. al., An efficient method to calculate excitation energy transfer in light-harvesting systems: application to the Fenna-Matthews-Olson complex, NJP 13 (2011) 113034 (and references therein)
    Extend ZOFE master equation to two-exciton space: unpublished
    Speed up ZOFE master equation: unblished

    """

    _BS = ham.shape[0]  # basis size
    _BSsq = _BS**2

    _numb_monomers = oop_shape[1]
    _numb_pm = oop_shape[0]

    rho = mat(rho_oop_vec[:_BSsq])  # ground state: rho[0,0], oop[pm,n,0,0] --> oop.shape = (numb_pm, numb_monomers, _BS, _BS)
    oop = tens(rho_oop_vec[_BSsq:], oop_shape)

    sum_oop = oop.sum(axis=0) #sum over pseudomode index

    # ZOFE master equation
    a_op = -1j*ham
    for n in np.arange(_numb_monomers):
        a_op -= np.dot(dag(L_n[n]), sum_oop[n])
        
    b_op = np.dot(a_op, rho)
    for n in np.arange(_numb_monomers):
        b_op += np.dot(np.dot(L_n[n], rho), dag(sum_oop[n]))

    rhodot = b_op + dag(b_op)
   
    # O operator evolution equation (uses a_op from above)
    oopdot = np.zeros_like(oop)
    for pm in np.arange(_numb_pm):
        for n in np.arange(_numb_monomers):
            oopdot[pm,n] = Gamma[pm,n]*L_n[n] - w[pm,n]*oop[pm,n] \
                + np.dot(a_op, oop[pm,n]) - np.dot(oop[pm,n], a_op)
            
                    
    return np.append(vec(rhodot), vec(oopdot))
#========================================================


