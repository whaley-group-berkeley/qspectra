"""
This module defines some useful constants for converting between different
units.
"""
from numpy import pi, sqrt, log

# multiply by this constant to convert from fs to linear cm^-1:
CM_FS_LINEAR = 2.99792458e-5
# multiply by this constant to convert from fs to angular cm^-1:
CM_FS = pi * 2 * CM_FS_LINEAR

# multiply by this constant to convert from degrees K to angular cm^-1,
# by implicitly inserting Boltzmann's constant:
CM_K = 0.69503476

# ratio between standard deviation and full-width-at-half-maximum for a
# gaussian function:
GAUSSIAN_SD_FWHM = 1.0 / (2 * sqrt(2 * log(2)))
