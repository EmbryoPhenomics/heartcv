import numpy as np
import math
from more_itertools import prepend

cimport cython
cimport numpy as np

DTYPE = np.uint8
ctypedef np.uint8_t DTYPE_t

@cython.boundscheck(False)
cpdef np.ndarray[DTYPE_t, ndim=3] _import_to_nparray(reader, int start, int stop, int step):
    cdef int x, y, i, length
    cdef np.ndarray[DTYPE_t, ndim=2] frame 

    gen = reader(start, stop, step)
    first = next(gen)
    x,y = first.shape
    gen = prepend(first, gen)

    length = math.floor((stop-start)/step)

    cdef np.ndarray[DTYPE_t, ndim=3] frames = np.ascontiguousarray(np.empty((length, x, y), dtype=DTYPE))

    for i,frame in enumerate(gen):
        frames[i,:] = frame[:]

    return frames
