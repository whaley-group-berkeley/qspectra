QSpectra: Nonlinear Spectroscopy for Molecular Aggregates
=========================================================

QSpectra is a Python package designed for efficient and flexible calculations of
non-linear spectroscopy signals. The core idea is a dynamical model interface
which allows for flexible composition of different dynamical models with different
spectroscopy methods, as long as the dynamical model is prescribed by some sort of 
linear differential equation.

To enable efficient calculations, all simulations are performed under the
rotating wave approximation. Furthermore, because the effective Hamiltonians
conserve the number of electronic excitations, each model only propagates within
the necessary fixed subspaces of Liouville subspace. Finally, response function
based methods can be calculated in the Heisenberg picture, which removes the need
for nested loops to calculate dynamics.

Although the QSpectra framework is written in Python modules for new dynamical
models may be written in a compiled language such as Fortran or C when necessary
for performance.

Features
--------

- Hamiltonians:
    - Electronic systems (under effective Hamiltonians within the Heitler-London approximation)
    - Vibronic systems (electronic systems with explicit vibrational modes)
- Dynamical models:
    - Unitary
    - Redfield theory (secular/non-secular)
    - Zeroth order functional expansion (ZOFE), *in progress*
    - Hierarchical equations of motion (HEOM), *on the roadmap*
    - Time non-local ME (special case of HEOM), *on the roadmap*
- Spectroscopy/simulation:
    - Free evolution (no field)
    - Equations of motion including fields, such as:
        + The density matrix following pump pulses
        + Equation-of-motion phase-matched-approach (EOM-PMA), *in progress*
    - Linear response based methods, such as:
        + Linear absorption
        + Impulsive probe pulses
    - Third-order response functions (in particular, for 2D spectroscopy)

Examples
--------

Example notebooks demonstrating the features of QSpectra are included in the
"examples" directory. They can also be browsed online *at some link that will be
inserted before the final release of this project*.
