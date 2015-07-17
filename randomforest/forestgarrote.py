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
        print weighted.shape
        print "non zero:", weighted.nnz, "of", (weighted.shape[0]*weighted.shape[1])

        # TODO: Train the Lasso.

        raise NotImplementedError

    def predict(self, data):
        """
        Predict the classes of the given data.

        :param data: the data
        :return: classes of the data
        """
        raise NotImplementedError
