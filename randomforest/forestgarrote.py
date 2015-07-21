import numpy
import sklearn
import sklearn.linear_model
import scipy
import scipy.sparse


class ForestGarrote(object):

    def __init__(self, rf):
        """
        Initialize the forest garrote.

        :param rf: the random forest
        """
        if len(rf.classes()) != 2:
            raise Exception("Currently, the forest garrote is only implemented for 2-class problems.")
        self._rf = rf

    def refine(self, data, labels):
        """
        Refine the stored random forest.

        :param data: the data
        :param labels: classes of the data
        """
        # Get the weighted node index vectors as new features.
        weighted = self._rf.weighted_index_vectors(data)

        # Translate the labels to 0 and 1.
        tmp_labels = numpy.zeros(labels.shape, dtype=numpy.float_)
        tmp_labels[numpy.where(labels == self._rf.classes()[1])] = 1.

        # Train the Lasso and save the best coefficients.
        # TODO: Find a better value for n_alphas.
        gram = weighted.transpose().dot(weighted)
        alphas, coefs, dual_gaps = sklearn.linear_model.lasso_path(weighted, tmp_labels, positive=True, n_alphas=100,
                                                                   precompute=True, Gram=gram)

        # Build the new forest by keeping all nodes on the path from the root to the nodes with non-zero weight.
        coefs = coefs[:, -1]
        nnz = coefs.nonzero()[0]
        nnz_coefs = coefs[nnz]
        print "creating sub forest"
        self._rf = self._rf.sub_fg_forest(nnz, nnz_coefs)
        print "done creating sub forest"

    def predict(self, data):
        """
        Predict the classes of the given data.

        :param data: the data
        :return: classes of the data
        """
        return self._rf.predict(data)
