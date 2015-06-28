import numpy
cimport numpy
cimport cython

DTYPE = numpy.int64
ctypedef numpy.int64_t DTYPE_t

INT = numpy.int
ctypedef numpy.int_t INT_t

@cython.boundscheck(False)
@cython.cdivision(True)
def find_best_gini(numpy.ndarray[DTYPE_t, ndim=1] arr, numpy.ndarray[DTYPE_t, ndim=1] priors):
    """
    Given an array with classes, find the split index where the gini is best.

    :param arr: array with classes
    :param priors: prior label count
    :param class_count: number of classes
    :return: best_gini, index
    """
    assert arr.dtype == DTYPE and priors.dtype == DTYPE

    cdef numpy.ndarray[DTYPE_t, ndim=1] counts = numpy.zeros((len(priors,)), dtype=DTYPE)
    cdef double count_left = 0
    cdef double count_right = arr.shape[0]
    cdef double count_total = count_right
    cdef double best_gini = -1
    cdef INT_t best_index = 0
    cdef INT_t i, j, c
    cdef DTYPE_t l
    cdef double gini_left, gini_right, p_left, p_right, gini
    for i in xrange(arr.shape[0]-1):
        l = arr[i]
        counts[l] += 1
        count_left += 1
        count_right -= 1

        gini_left = 1.0
        gini_right = 1.0
        for j in xrange(counts.shape[0]):
            c = counts[j]
            p_left = c / count_left
            p_right = (priors[j] - c) / count_right
            gini_left -= p_left*p_left
            gini_right -= p_right*p_right
        gini = count_left*gini_left + count_right*gini_right

        if best_gini < 0:
            best_gini = gini
            best_index = i
        else:
            if gini < best_gini:
                best_gini = gini
                best_index = i

    return best_gini, best_index+1
