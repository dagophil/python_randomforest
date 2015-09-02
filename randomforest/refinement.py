import sklearn.svm
import sklearn.grid_search
import numpy


def global_refinement(rf, data, labels, n_jobs=None):
    """
    Apply the global refinement from the paper "Global Refinement of Random Forest".

    :param rf: the random forest
    :param data: the data
    :param labels: classes for the data
    :param n_jobs: number of parallel jobs
    :return: refined random forest
    """
    if len(rf.classes()) != 2:
        raise Exception("Currently, the forest garrote is only implemented for 2-class problems.")
    if n_jobs is None:
        n_jobs = -1

    # Get the leaf index vectors as new features.
    leaf_ids = rf.leaf_index_vectors(data)

    # Do a cross validated grid search to find the optimal parameter C for the SVM.
    svm = sklearn.svm.LinearSVC(penalty="l1", dual=False)
    c_candidates = numpy.logspace(-5, 4, 30)
    gs = sklearn.grid_search.GridSearchCV(estimator=svm, param_grid=dict(C=c_candidates), n_jobs=n_jobs)
    gs.fit(leaf_ids, labels)

    # Do not use the C with the best score, but the smallest C with a score >= 0.9*best_score.
    c_candidates = [x.parameters["C"] for x in gs.grid_scores_ if x.mean_validation_score >= 0.9 * gs.best_score_]
    c_candidates.sort()
    best_c = c_candidates[0]

    # Train the SVM and get the coefficients.
    svm = sklearn.svm.LinearSVC(penalty="l1", dual=False, C=best_c)
    svm.fit(leaf_ids, labels)
    coefs = svm.coef_[0, :]

    # Build the new forest.
    nnz = coefs.nonzero()[0]
    nnz_coefs = coefs[nnz]
    return rf.sub_gr_forest(nnz, nnz_coefs)
