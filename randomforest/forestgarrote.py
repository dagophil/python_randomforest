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
        # x = scipy.sparse.
        ids = self._rf._trees[0].node_index_vectors(data)
        print ids.shape


        raise NotImplementedError

    def predict(self, data):
        """
        Predict the classes of the given data.

        :param data: the data
        :return: classes of the data
        """
        raise NotImplementedError
