import sklearn.svm


def global_refinement(rf, data, labels):
    """
    Apply the global refinement from the paper "Global Refinement of Random Forest".

    :param rf: the random forest
    :param data: the data
    :param labels: classes for the data
    :return: refined random forest
    """
    if len(rf.classes()) != 2:
        raise Exception("Currently, the forest garrote is only implemented for 2-class problems.")

    # Get the leaf index vectors as new features.
    leaf_ids = rf.leaf_index_vectors(data)

    # Train the SVM.
    svm = sklearn.svm.LinearSVC(penalty="l1", dual=False, C=0.01)
    svm.fit(leaf_ids, labels)
    coefs = svm.coef_[0, :]

    # Build the new forest.
    nnz = coefs.nonzero()[0]
    nnz_coefs = coefs[nnz]
    return rf.sub_gr_forest(nnz, nnz_coefs)
