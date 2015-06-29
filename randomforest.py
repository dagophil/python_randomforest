import numpy
import networkx
import collections
import random
import gini
import time


class DecisionTreeClassifier(object):

    def __init__(self, n_rand_dims="all"):
        self._n_rand_dims = n_rand_dims
        self._graph = networkx.DiGraph()
        self._label_names = None

    def _find_n_rand_dims(self, sh):
        """
        Given the shape of the training data, return the number of considered feature dimensions.

        :param sh: shape
        :return: number of considered feature dimensions
        """
        if self._n_rand_dims == "all":
            n = sh[1]
        elif self._n_rand_dims == "auto_reduced":
            n = numpy.sqrt(sh[1])
        else:
            n = self._n_rand_dims
        n = int(numpy.ceil(n))
        n = min(n, sh[1])
        return n

    def fit(self, data, labels):
        """
        Train a decision tree.

        :param data: the data
        :param labels: classes of the data
        """
        assert data.shape[0] == labels.shape[0]

        # Translate the labels to 0, 1, 2, ...
        self._label_names, labels, label_counts = numpy.unique(labels, return_inverse=True, return_counts=True)

        # Get the number of feature dimensions that are considered in each split.
        n_rand_dims = self._find_n_rand_dims(data.shape)
        dims = range(data.shape[1])

        # Create the index vector.
        instances = numpy.array(xrange(0, data.shape[0]))

        # Add the root node to the graph and the queue.
        self._graph.add_node(0, begin=0, end=data.shape[0], label_names=self._label_names, label_count=label_counts)
        next_node_id = 1
        qu = collections.deque()
        qu.append(0)

        # Split each node.
        while len(qu) > 0:
            node_id = qu.popleft()
            node = self._graph.node[node_id]

            # Do not split if there is only one label left in the node.
            if len(node["label_names"]) <= 1:
                continue

            begin = node["begin"]
            end = node["end"]
            node_instances = instances[begin:end]
            label_priors = node["label_count"]

            # Find the best split.
            best_gini = -1  # gini is always >= 0, so we can use -1 as "not set" value
            best_index = 0
            best_dim = 0
            split_dims = random.sample(dims, n_rand_dims)
            for d in split_dims:
                feats = data[node_instances, d]
                sorted_instances = numpy.argsort(feats)
                sorted_labels = labels[node_instances[sorted_instances]]

                gini_value, index = gini.find_best_gini(sorted_labels, label_priors)

                if best_gini < 0:
                    best_gini = gini_value
                    best_index = index
                    best_dim = d
                else:
                    if gini_value < best_gini:
                        best_gini = gini_value
                        best_index = index
                        best_dim = d

            # Sort the index vector of the current node, so that the instances of the left child are in the left half.
            feats = data[node_instances, best_dim]
            sorted_instances = numpy.argsort(feats)
            instances[begin:end] = node_instances[sorted_instances]

            # Get the label count of each child.
            middle = begin+best_index
            labels_left = labels[instances[begin:middle]]
            labels_right = labels[instances[middle:end]]
            cl_left, counts_left = numpy.unique(labels_left, return_counts=True)
            cl_right, counts_right = numpy.unique(labels_right, return_counts=True)

            if len(cl_left) > 1 and len(cl_right) > 1:
                # Add the children to the graph.
                self._graph.add_node(next_node_id, begin=begin, end=middle,
                                     label_names=cl_left, label_count=counts_left)
                self._graph.add_node(next_node_id+1, begin=middle, end=end,
                                     label_names=cl_right, label_count=counts_right)
                self._graph.add_edge(node_id, next_node_id)
                self._graph.add_edge(node_id, next_node_id+1)
                qu.append(next_node_id)
                qu.append(next_node_id+1)
                next_node_id += 2

                # Update the node with the split information.
                node["split_dim"] = best_dim
                node["split_value"] = (data[middle-1, best_dim] + data[middle, best_dim]) / 2.0

    def predict_proba(self, data):
        """
        Predict the class probabilities of the data.

        :param data: the data
        :return: class probabilities of the data
        """
        raise NotImplementedError

    def predict(self, data):
        """
        Predict classes of the data.

        :param data: the data
        :return: classes of the data
        """
        raise NotImplementedError


class RandomForestClassifier(object):

    def __init__(self, n_estimators=10, n_rand_dims="auto"):
        if n_rand_dims == "auto":
            tree_rand_dims = "auto_reduced"
        else:
            tree_rand_dims = "all"

        self._n_estimators = n_estimators
        self._trees = [DecisionTreeClassifier(n_rand_dims=tree_rand_dims) for _ in xrange(n_estimators)]

    def fit(self, data, labels):
        """
        Train a random forest.

        :param data: the data
        :param labels: classes of the data
        """
        raise NotImplementedError

    def predict_proba(self, data):
        """
        Predict the class probabilities of the data.

        :param data: the data
        :return: class probabilities of the data
        """
        raise NotImplementedError

    def predict(self, data):
        """
        Predict classes of the data.

        :param data: the data
        :return: classes of the data
        """
        raise NotImplementedError
