import numpy
import sklearn
import sklearn.linear_model


def forest_garrote(rf, data, labels, group_size=None):
    """
    Apply the forest garrote on the given random forest.

    :param rf: the random forest
    :param data: the data
    :param labels: classes for the data
    :param group_size: size of each Lasso group
    :return: refined random forest
    """
    if len(rf.classes()) != 2:
        raise Exception("Currently, the forest garrote is only implemented for 2-class problems.")

    # Get the weighted node index vectors as new features.
    weighted = rf.weighted_index_vectors(data)

    # Translate the labels to 0 and 1.
    tmp_labels = numpy.zeros(labels.shape, dtype=numpy.float_)
    tmp_labels[numpy.where(labels == rf.classes()[1])] = 1.

    # TODO: Find a better value for n_alphas.
    if group_size is None:
        # Train the Lasso on the whole forest.
        gram = weighted.transpose().dot(weighted)
        alphas, coefs, dual_gaps = sklearn.linear_model.lasso_path(weighted, tmp_labels, positive=True, n_alphas=100,
                                                                   precompute=True, Gram=gram)
        coefs = coefs[:, -1]
    else:
        # Make tree groups of the given size and train a Lasso on each group.
        print len(rf._trees)
        print group_size

        raise NotImplementedError

    # Build the new forest by keeping all nodes on the path from the root to the nodes with non-zero weight.
    nnz = coefs.nonzero()[0]
    nnz_coefs = coefs[nnz]
    return rf.sub_fg_forest(nnz, nnz_coefs)
