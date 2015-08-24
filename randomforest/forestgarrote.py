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
    weighted = rf.weighted_index_vectors(data).tocsc()

    # Transform the labels to 0 and 1.
    tmp_labels = numpy.zeros(labels.shape, dtype=numpy.float_)
    tmp_labels[numpy.where(labels == rf.classes()[1])] = 1.

    # TODO: Find a better value for n_alphas.
    if group_size is None:
        # Train the Lasso on the whole forest.
        gram = weighted.transpose().dot(weighted)
        alphas, coefs, dual_gaps = sklearn.linear_model.lasso_path(weighted, tmp_labels, positive=True, n_alphas=100,
                                                                   precompute=True, Gram=gram)

        coefs = coefs[:, -1]

        # Build the new forest.
        nnz = coefs.nonzero()[0]
        nnz_coefs = coefs[nnz]
        return rf.sub_fg_forest(nnz, nnz_coefs, rf.num_trees())

    else:
        # Create the tree groups and find the number of nodes in each group.
        n_groups = rf.num_trees() / group_size
        group_sizes = [group_size] * n_groups
        for i in xrange(rf.num_trees() - n_groups*group_size):
            group_sizes[i] += 1
        group_slices = numpy.cumsum([0]+group_sizes)
        n_nodes = [sum([rf.num_nodes(j) for j in xrange(group_slices[i], group_slices[i+1])]) for i in xrange(n_groups)]
        node_slices = numpy.cumsum([0]+n_nodes)

        # Train a Lasso for each group.
        coef_list = []
        for i in xrange(n_groups):
            print "Computing Lasso for group", i+1, "of", n_groups
            sub_weights = weighted[:, node_slices[i]:node_slices[i+1]]
            gram = sub_weights.transpose().dot(sub_weights)
            alphas, coefs, dual_gaps = sklearn.linear_model.lasso_path(sub_weights, tmp_labels, positive=True,
                                                                       n_alphas=100, precompute=True, Gram=gram)
            coef_list.append(coefs[:, -1])
        coefs = numpy.concatenate(coef_list)

        # Build the new forest by keeping all nodes on the path from the root to the nodes with non-zero weight.
        nnz = coefs.nonzero()[0]
        nnz_coefs = coefs[nnz]
        return rf.sub_fg_forest(nnz, nnz_coefs, group_size)
