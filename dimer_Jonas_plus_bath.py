#!/usr/bin/env python

#started: 08/14/13
#last modified: 08/14/13

# Calculate absorption spectrum of dimer.
# Each monomer has one harmonic vibrational mode damped by harmonic Markovian bath.
#
# Checks:
# 1) This script reproduced, using the Lindblad master equation propagation corresponding to Chen et al JChemPhys 131, 094502 (2009), the four different one-pseudomode zero temperature monomer absorption spectra of our PM paper perfectly (JR20111018).
# This shows that the Lindblad master equation propagation for one pseudomode at zero temperature is equivalent to the Markovian QSD propagation (and the NMQSD propagation) used in our PM paper.

# Modules
from qutip import *
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os, sys

# Parameters
NA = 7 # number of vibrational levels in system A
NB = 7 # number of vibrational levels in system B
omegaA = 0.0200 # mode frequency of the vibrations in system A
omegaB = 0.0200 # mode frequency of the vibrations in system B
EA = 1.1400 # electronic energy of excited state in system A (we assume that the electronic energy of the ground state is zero)
EB = 1.1600 # electronic energy of excited state in system B (we assume that the electronic energy of the ground state is zero)
ccA = 0.0030 # coupling constant between the electronic and vibrational subsystems in system A (=\sqrt{\hbar/2}*\omega_A*d_A)
ccB = 0.0030 # coupling constant between the electronic and vibrational subsystems in system B (=\sqrt{\hbar/2}*\omega_B*d_B)
J=0.0066 # coupling between dimer electronic singly excited states
gamma = 0.1 # mode dissipation rate due to coupling with the bath (QuTip gamma); QuTip gamma = 2* our_gamma in our PM paper (compare QuTiP arXiv:1110.0573v1 paper (4 Oct 2011), Eq.(9)  and  Jan PROM I, page 159, Eq.(1))
n_th_b = 0. # expectation value for number of quanta in thermal state of vibrational mode

# States and operators
g = basis(3,0) # ground electronic state (same as fock(2,0))
A = basis(3,1) # singly excited state in which monomer A is excited and monomer B is not
B = basis(3,2) # singly excited state in which monomer B is excited and monomer A is not
g_proj = tensor(g*g.dag(), qeye(NA), qeye(NB)) # projector onto ground electronic state
A_proj = tensor(A*A.dag(), qeye(NA), qeye(NB)) # projector onto state A
B_proj = tensor(B*B.dag(), qeye(NA), qeye(NB)) # projector onto state B
aA  = tensor(qeye(3), destroy(NA), qeye(NB)) #annihilation operator of vibrational mode in electronic ground state for monomer A
aB  = tensor(qeye(3), qeye(NA), destroy(NB)) #annihilation operator of vibrational mode in electronic ground state for monomer B

# Hamiltonian
H = EA*A_proj + EB*B_proj + tensor(J*(A*B.dag() + B*A.dag()), qeye(NA), qeye(NB)) \
    - ccA*A_proj*(aA + aA.dag()) - ccB*B_proj*(aB + aB.dag()) + omegaA*aA.dag()*aA + omegaB*aB.dag()*aB

# Lindblad operators (collapse operators)
# The QuTiP collapse operators C (see QuTiP arXiv:1110.0573v1 paper (4 Oct 2011), Eq.(9)) are exactly the Lindblad operators L in Walters Habil (see Walters Habil, page Eq.(1.2))
c_op_list = []

L_1 = aA

rate = gamma * (1 + n_th_b) #be careful here with temperature (n_th_b > 0), because n_th is only for pseudomode b. Unclear yet how to implement it for "molecular mode".
if rate > 0.0:
    c_op_list.append(sqrt(rate) * L_1)

rate = gamma * n_th_b
if rate > 0.0:
    c_op_list.append(sqrt(rate) * L_1.dag())

L_2 = aB

rate = gamma * (1 + n_th_b) #be careful here with temperature (n_th_b > 0), because n_th is only for pseudomode b. Unclear yet how to implement it for "molecular mode".
if rate > 0.0:
    c_op_list.append(sqrt(rate) * L_2)

rate = gamma * n_th_b
if rate > 0.0:
    c_op_list.append(sqrt(rate) * L_2.dag())

#

rho0 = tensor(A*A.dag(), thermal_dm(NA,1), thermal_dm(NB,1)) #ket2dm(basis(NA,0)), ket2dm(basis(NB,0)))
psi0 = tensor(A, basis(NA,0), basis(NB,0))

tlist = np.linspace(0, 400, 1000)

rho_list = odesolve(H, rho0, tlist, c_op_list, [A_proj,B_proj])
psi_list = odesolve(H, psi0, tlist, [], [A_proj,B_proj])

plt.plot(tlist,np.real(psi_list[0]),tlist,np.real(psi_list[1]))
plt.xlabel('Time')
plt.ylabel('Populations of A and B')
plt.show()

plt.plot(tlist,np.real(rho_list[0]),tlist,np.real(rho_list[1]))
plt.xlabel('Time')
plt.ylabel('Populations of A and B')
plt.show()