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
        self._coefs = None

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
        gram = weighted.transpose().dot(weighted)
        alphas, coefs, dual_gaps = sklearn.linear_model.lasso_path(weighted, tmp_labels, positive=True, n_alphas=100,
                                                                   precompute=True, Gram=gram)
        self._coefs = scipy.sparse.diags(coefs[:, -1], 0)

        # nnz = len(self._coefs.nonzero()[0])
        # print nnz, "of", weighted.shape[1], "weights are non-zero (%f%%)" % (nnz/float(weighted.shape[1]))

        # TODO: Find a better value for n_alphas.

    def predict(self, data):
        """
        Predict the classes of the given data.

        :param data: the data
        :return: classes of the data
        """
        # Get the weighted node index vectors.
        weighted = self._rf.weighted_index_vectors(data)

        # Multiply with the lasso weights.
        weighted = weighted * self._coefs

        # Round the values.
        val = numpy.array(weighted.sum(axis=1)).flatten()
        return numpy.round(val).astype(numpy.uint8)
