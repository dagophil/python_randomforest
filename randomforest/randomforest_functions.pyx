#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

import numpy
cimport numpy
cimport cython
import scipy.sparse

FLOAT = numpy.float_
ctypedef numpy.float_t FLOAT_t

INT = numpy.int_
ctypedef numpy.int_t INT_t

UINT8 = numpy.uint8
ctypedef numpy.uint8_t UINT8_t


def find_best_gini(numpy.ndarray[INT_t, ndim=1] arr, numpy.ndarray[INT_t, ndim=1] priors):
    """
    Given an array with classes, find the split index where the gini is best.

    :param arr: array with classes
    :param priors: prior label count
    :return: best_gini, index, split_found
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
    cdef bint split_found = False
    for i in xrange(arr.shape[0]-1):
        l = arr[i]
        counts[l] += 1
        count_left += 1
        count_right -= 1

        # Skip if there is no new split.
        if l == arr[i+1]:
            continue

        split_found = True
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

    return best_gini, best_index+1, split_found


def leaf_ids(numpy.ndarray[FLOAT_t, ndim=2] data, numpy.ndarray[INT_t, ndim=2] children,
             numpy.ndarray[INT_t, ndim=1] split_dims, numpy.ndarray[FLOAT_t, ndim=1] split_values):
    """
    Find the leaf index of each instance in data.

    :param data: the data
    :param children: child information of the graph
    :param split_dims: node split dimensions
    :param split_values: node split values
    :return: leaf indices of the given data
    """
    cdef numpy.ndarray[INT_t, ndim=1] indices = numpy.zeros((data.shape[0],), dtype=INT)
    cdef INT_t i, node

    for i in xrange(data.shape[0]):
        node = 0
        while children[node, 0] >= 0:
            if data[i, split_dims[node]] < split_values[node]:
                node = children[node, 0]
            else:
                node = children[node, 1]
        indices[i] = node

    return indices


def predict_proba(numpy.ndarray[FLOAT_t, ndim=2] data, numpy.ndarray[INT_t, ndim=2] children,
                  numpy.ndarray[INT_t, ndim=1] split_dims, numpy.ndarray[FLOAT_t, ndim=1] split_values,
                  numpy.ndarray[FLOAT_t, ndim=2] label_probs):
    """
    Predict the class probabilities of the given data.

    :param data: the data
    :param children: child information of the graph
    :param split_dims: node split dimensions
    :param split_values: node split values
    :param label_probs: label probabilities in each node
    :return: class probabilities of the data
    """
    cdef numpy.ndarray[FLOAT_t, ndim=2] probs = numpy.zeros((data.shape[0], label_probs.shape[1]), dtype=FLOAT)
    cdef INT_t i, j, node
    cdef FLOAT_t s

    cdef numpy.ndarray[INT_t, ndim=1] indices = leaf_ids(data, children, split_dims, split_values)

    for i in xrange(data.shape[0]):
        node = indices[i]
        for j in xrange(label_probs.shape[1]):
            probs[i, j] = label_probs[node, j]

    return probs


def node_ids(numpy.ndarray[FLOAT_t, ndim=2] data, numpy.ndarray[INT_t, ndim=2] children,
             numpy.ndarray[INT_t, ndim=1] split_dims, numpy.ndarray[FLOAT_t, ndim=1] split_values):
    """
    Return the node index vector of each instance in data.

    :param data: the data
    :param children: child information of the graph
    :param split_dims: node split dimensions
    :param split_values: node split values
    :return: node index vectors (shape data.shape[0] x num_nodes, value is 1 if instance is in node else 0)
    """
    cdef numpy.ndarray[UINT8_t, ndim=2] indices = numpy.zeros((data.shape[0], children.shape[0]), dtype=UINT8)
    cdef INT_t i, node

    for i in xrange(data.shape[0]):
        node = 0
        indices[i, node] = 1
        while children[node, 0] >= 0:
            if data[i, split_dims[node]] < split_values[node]:
                node = children[node, 0]
            else:
                node = children[node, 1]
            indices[i, node] = 1
    return indices


def node_ids_sparse(numpy.ndarray[FLOAT_t, ndim=2] data, numpy.ndarray[INT_t, ndim=2] children,
                    numpy.ndarray[INT_t, ndim=1] split_dims, numpy.ndarray[FLOAT_t, ndim=1] split_values,
                    INT_t tree_depth):
    """
    Return the node index vector of each instance in data as a sparse matrix.

    :param data: the data
    :param children: child information of the graph
    :param split_dims: node split dimensions
    :param split_values: node split values
    :param tree_depth: the tree depth, so an estimate of the number of non-zero entries can be computed
    :return: node index vectors (shape data.shape[0] x num_nodes, value is 1 if instance is in node else 0)
    """
    cdef INT_t count_nonzero = data.shape[0] * tree_depth
    cdef numpy.ndarray[INT_t, ndim=1] rows = numpy.zeros(count_nonzero, dtype=INT)
    cdef numpy.ndarray[INT_t, ndim=1] cols = numpy.zeros(count_nonzero, dtype=INT)
    cdef numpy.ndarray[UINT8_t, ndim=1] vals = numpy.zeros(count_nonzero, dtype=UINT8)  # this should be boolean, but cython does not support bool arrays
    cdef INT_t i, next, node

    next = 0
    for i in xrange(data.shape[0]):
        node = 0
        rows[next] = i
        cols[next] = node
        vals[next] = 1
        next += 1
        while children[node, 0] >= 0:
            if data[i, split_dims[node]] < split_values[node]:
                node = children[node, 0]
            else:
                node = children[node, 1]
            rows[next] = i
            cols[next] = node
            vals[next] = 1
            next += 1

    return scipy.sparse.coo_matrix((vals[:next], (rows[:next], cols[:next])), shape=(data.shape[0], children.shape[0]))


def adjusted_node_weights(numpy.ndarray[INT_t, ndim=2] children, numpy.ndarray[FLOAT_t, ndim=2] label_probs):
    """
    Return the adjusted node weights.

    :param children: child information of the graph
    :param label_probs: label probabilities in each node
    :return: adjusted node weights
    """
    assert label_probs.shape[1] == 2
    assert label_probs.shape[0] == children.shape[0]

    cdef numpy.ndarray[FLOAT_t, ndim=1] weights = numpy.zeros((children.shape[0],), dtype=FLOAT)
    cdef INT_t i
    cdef FLOAT_t s

    weights[0] = label_probs[0, 1]
    for i in xrange(children.shape[0]):
        left = children[i, 0]
        if left >= 0:
            weights[left] = label_probs[left, 1] - label_probs[i, 1]
        right = children[i, 1]
        if right >= 0:
            weights[right] = label_probs[right, 1] - label_probs[i, 1]

    return weights


def weighted_node_ids(numpy.ndarray[FLOAT_t, ndim=2] data, numpy.ndarray[INT_t, ndim=2] children,
                      numpy.ndarray[INT_t, ndim=1] split_dims, numpy.ndarray[FLOAT_t, ndim=1] split_values,
                      numpy.ndarray[FLOAT_t, ndim=2] label_probs):
    """
    Return the weighted node index vector of each instance in data.

    :param data: the data
    :param children: child information of the graph
    :param split_dims: node split dimensions
    :param split_values: node split values
    :param label_probs: label probabilities in each node
    :return: weighted node index vector
    """
    assert label_probs.shape[1] == 2
    assert label_probs.shape[0] == children.shape[0]

    cdef numpy.ndarray[FLOAT_t, ndim=2] weights = numpy.zeros((data.shape[0], children.shape[0]), dtype=FLOAT)
    cdef INT_t i, node, next_node
    cdef FLOAT_t s

    for i in xrange(data.shape[0]):
        node = 0
        weights[i, node] = label_probs[node, 1]
        while children[node, 0] >= 0:
            if data[i, split_dims[node]] < split_values[node]:
                next_node = children[node, 0]
            else:
                next_node = children[node, 1]
            weights[i, next_node] = label_probs[next_node, 1] - label_probs[node, 1]
            node = next_node

    return weights


def weighted_node_ids_sparse(numpy.ndarray[FLOAT_t, ndim=2] data, numpy.ndarray[INT_t, ndim=2] children,
                             numpy.ndarray[INT_t, ndim=1] split_dims, numpy.ndarray[FLOAT_t, ndim=1] split_values,
                             numpy.ndarray[FLOAT_t, ndim=2] label_probs, INT_t tree_depth):
    """
    Return the weighted node index vector of each instance in data as a sparse matrix.

    :param data: the data
    :param children: child information of the graph
    :param split_dims: node split dimensions
    :param split_values: node split values
    :param label_probs: label counts in each node
    :param tree_depth: the tree depth, so an estimate of the number of non-zero entries can be computed
    :return: weighted node index vector
    """
    assert label_probs.shape[1] == 2
    assert label_probs.shape[0] == children.shape[0]

    cdef INT_t count_nonzero = data.shape[0] * tree_depth
    cdef numpy.ndarray[INT_t, ndim=1] rows = numpy.zeros(count_nonzero, dtype=INT)
    cdef numpy.ndarray[INT_t, ndim=1] cols = numpy.zeros(count_nonzero, dtype=INT)
    cdef numpy.ndarray[FLOAT_t, ndim=1] vals = numpy.zeros(count_nonzero, dtype=FLOAT)
    cdef INT_t i, next, node, next_node
    cdef FLOAT_t s

    next = 0
    for i in xrange(data.shape[0]):
        node = 0
        rows[next] = i
        cols[next] = node
        vals[next] = label_probs[node, 1]
        next += 1
        while children[node, 0] >= 0:
            if data[i, split_dims[node]] < split_values[node]:
                next_node = children[node, 0]
            else:
                next_node = children[node, 1]
            rows[next] = i
            cols[next] = next_node
            vals[next] = label_probs[next_node, 1] - label_probs[node, 1]
            next += 1
            node = next_node

    return scipy.sparse.coo_matrix((vals[:next], (rows[:next], cols[:next])), shape=(data.shape[0], children.shape[0]))
