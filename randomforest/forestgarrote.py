class ForestGarrote(object):

    def __init__(self, rf):
        """
        Initialize the forest garrote.

        :param rf: the random forest
        """
        self._rf = rf

    def refine(self):
        """
        Refine the stored random forest.
        """
        pass

    def predict(self, data):
        """
        Predict the classes of the given data.

        :param data: the data
        :return: classes of the data
        """
        raise NotImplementedError
