import numpy
import sklearn.linear_model


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

        las = sklearn.linear_model.Lasso()
        alphas, coefs, dual_gaps = las.path(weighted, tmp_labels, positive=True)

        # TODO: Use the coefs to improve the weights and merge leaves.

        raise NotImplementedError

    def predict(self, data):
        """
        Predict the classes of the given data.

        :param data: the data
        :return: classes of the data
        """
        raise NotImplementedError
