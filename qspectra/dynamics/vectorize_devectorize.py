#!/usr/bin/env python


import numpy as np


def vec(tensor):
    """Make vector out of tensor of arbitrary rank."""
    return tensor.reshape((-1), order='F')


def mat(vec):
    """Make square matrix out of vector."""
    _N = int(np.sqrt(vec.size))
    return vec.reshape((_N, _N), order='F')

def tens(vec, shape):
    """Make tensor of defined shape out of vector.
    Equivalent to Fortran reshape(vec,shape) function.
    JR20111026.

    Example:
    >>> import numpy as np
    >>> import vectorize_devectorize
    >>> a=np.array([[[1,2],[2,4],[9,9]],[[5,6],[7,6],[8,8]]])
    >>> print a
    [[[1 2]
      [2 4]
      [9 9]]

     [[5 6]
      [7 6]
      [8 8]]]
    >>> b=nj.vec(a)
    >>> print b
    [1 5 2 7 9 8 2 6 4 6 9 8]
    >>> c=nj.tens(b,a.shape)
    >>> a.all()==c.all()
    True

    """
    return vec.reshape(shape, order='F')





