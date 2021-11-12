QSpectra: Nonlinear Spectroscopy for Molecular Aggregates
=========================================================

QSpectra is a Python package designed for fast and flexible calculations of non-
linear spectroscopy signals on molecular aggregates, such as photosynthetic
light-harvesting complexes. The focus is on solving approximate models of
electronic dynamics under known effective Hamiltonians as open quantum systems.

The core idea is a dynamical model interface which allows for flexible
composition of different dynamical models with different spectroscopy methods,
as long as the equation of motion is a linear function of the system
Hamiltonian.

To enable efficient calculations, all simulations are performed under the
rotating wave approximation. Furthermore, because the effective Hamiltonians
conserve the number of electronic excitations, each model (when practical) only
propagates within the necessary fixed subspaces of Liouville subspace.

Although the QSpectra framework is written in Python, we expect that eventually
submodules for new dynamical models may be written in a compiled language such
as Fortran or C when if necessary for satisfactory performance.

Install
-------

First, make sure you're running Python 2.7 and have recent versions of numpy
and scipy installed.

Then, clone the git repository and use `pip`:
```
git clone https://github.com/whaley-group-berkeley/QSpectra.git
pip install -e QSpectra
```

I highly recommend using `-e` flag, which keeps the install directory in-place
for local development.

To view the example notebooks, you need ``jupyter``. To run the unit tests,
you need ``nosetests``.

Features
--------

- Hamiltonians:
    - Electronic systems (under effective Hamiltonians within the Heitler-London
      approximation)
    - Vibronic systems (electronic systems with explicit vibrational modes)
- Dynamical models:
    - Unitary
    - Redfield theory (secular/non-secular)
    - Zeroth order functional expansion (ZOFE)
    - Hierarchical equations of motion (HEOM).
    - Time non-local ME (special case of HEOM), *on the roadmap*
- Spectroscopy/simulation:
    - Free evolution (no field)
    - Equations of motion including fields, such as:
        + The density matrix following pump pulses
        + Equation-of-motion phase-matched-approach (EOM-PMA), *in progress*
    - Linear response based methods, such as:
        + Absorption and emission spectra
        + Impulsive probe pulses
    - Third-order response functions (in particular, for 2D spectroscopy)

Examples
--------

Example notebooks demonstrating the features of QSpectra are included in the
"examples" directory.
