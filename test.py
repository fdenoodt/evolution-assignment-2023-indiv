import numpy
from numba import int32, jit

@jit(nopython=True)
def fast_function():
    return np.zeros(5, dtype=int32)

fast_function()