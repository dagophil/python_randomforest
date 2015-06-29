import numpy
cimport numpy
cimport cython

FLOAT = numpy.float_
ctypedef numpy.float_t FLOAT_t

INT = numpy.int_
ctypedef numpy.int_t INT_t


@cython.boundscheck(False)
@cython.cdivision(True)
def find_best_gini(numpy.ndarray[INT_t, ndim=1] arr, numpy.ndarray[INT_t, ndim=1] priors):
    """
    Given an array with classes, find the split index where the gini is best.

    :param arr: array with classes
    :param priors: prior label count
    :param class_count: number of classes
    :return: best_gini, index
    """
    assert arr.dtype == INT and priors.dtype == INT

    cdef numpy.ndarray[INT_t, ndim=1] counts = numpy.zeros((len(priors,)), dtype=INT)
    cdef double count_left = 0
    cdef double count_right = arr.shape[0]
    cdef double count_total = count_right
    cdef double best_gini = -1
    cdef INT_t best_index = 0
    cdef INT_t i, j, c
    cdef INT_t l
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


@cython.boundscheck(False)
@cython.cdivision(True)
def predict_proba(numpy.ndarray[FLOAT_t, ndim=2] data, numpy.ndarray[INT_t, ndim=2] children,
                  numpy.ndarray[INT_t, ndim=1] split_dims, numpy.ndarray[FLOAT_t, ndim=1] split_values,
                  numpy.ndarray[INT_t, ndim=2] label_count):
    """
    Predict the class probabilities of the given data.

    :param data: the data
    :param children: child information of the graph
    :param split_dims: node split dimensions
    :param split_values: node split values
    :param label_count: label counts in each node
    :return: class probabilities of the data
    """
    assert data.dtype == FLOAT and children.dtype == INT and split_dims.dtype == INT and split_values.dtype == FLOAT \
           and label_count.dtype == INT

    cdef numpy.ndarray[FLOAT_t, ndim=2] probs = numpy.zeros((data.shape[0], label_count.shape[1]), dtype=FLOAT)
    cdef numpy.ndarray[INT_t, ndim=1] label_sums = numpy.zeros((label_count.shape[0],), dtype=INT)

    cdef INT_t i, j, node
    cdef FLOAT_t s

    for i in xrange(label_count.shape[0]):
        for j in xrange(label_count.shape[1]):
            label_sums[i] += label_count[i, j]

    for i in xrange(data.shape[0]):
        node = 0
        while children[node, 0] >= 0:
            if data[i, split_dims[node]] < split_values[node]:
                node = children[node, 0]
            else:
                node = children[node, 1]
        s = float(label_sums[node])
        for j in xrange(label_count.shape[1]):
            probs[i, j] = label_count[node, j] / s

    return probs
