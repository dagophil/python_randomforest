import numpy
import sklearn.linear_model
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

        # TODO: Use sparse Lasso instead.
        # TODO: Use parameter coef_init for the weights and the real index vectors instead of the weighted ones.

        gram = weighted.transpose().dot(weighted)
        alphas, coefs, dual_gaps = sklearn.linear_model.lasso_path(weighted, tmp_labels, positive=True,
                                                                   precompute=True, Gram=gram)

        # TODO: Use more iterations.
        # TODO: Save coefs and use them in the prediction.

        raise NotImplemented

    def predict(self, data):
        """
        Predict the classes of the given data.

        :param data: the data
        :return: classes of the data
        """
        index_data = scipy.sparse.csc_matrix(self._rf.node_index_vectors(data))
        node_weights = self._rf.adjusted_node_weights()

        # TODO: Multiply the weights by the saved lasso coefficients.

        node_weights = scipy.sparse.diags(node_weights, 0)
        weighted_data = index_data*node_weights
        return numpy.round(weighted_data.sum(axis=1)/self._rf.num_trees()).astype(numpy.uint8)
