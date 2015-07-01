import numpy
import networkx
import collections
import random
import randomforest_functions
import time
import concurrent.futures
import multiprocessing


class DecisionTreeClassifier(object):

    def __init__(self, n_rand_dims="all"):
        self._n_rand_dims = n_rand_dims
        self._graph = networkx.DiGraph()
        self._label_names = None

    def classes(self):
        """
        Return the classes that were found in training.

        :return: the classes
        """
        return numpy.array(self._label_names)

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

    def fit(self, data, labels, bootstrap_sampling=False):
        """
        Train a decision tree.

        :param data: the data
        :param labels: classes of the data
        """
        assert data.shape[0] == labels.shape[0]

        # Translate the labels to 0, 1, 2, ...
        self._label_names, labels, label_counts = numpy.unique(labels, return_inverse=True, return_counts=True)
        assert len(self._label_names) > 1
        label_names = numpy.array(xrange(len(self._label_names)))

        # Get the number of feature dimensions that are considered in each split.
        n_rand_dims = self._find_n_rand_dims(data.shape)
        dims = range(data.shape[1])

        # Create the index vector.
        instances = numpy.array(xrange(0, data.shape[0]))
        # if bootstrap_sampling:
        #     sample_instances = numpy.random.random_integers(0, data.shape[0]-1, data.shape[0])
        # else:
        #     sample_instances = numpy.array(instances)

        # Add the root node to the graph and the queue.
        self._graph.add_node(0, begin=0, end=data.shape[0], num_labels=len(label_names), label_counts=label_counts)
        next_node_id = 1
        qu = collections.deque()
        qu.append(0)

        # Split each node.
        while len(qu) > 0:
            node_id = qu.popleft()
            node = self._graph.node[node_id]
            begin = node["begin"]
            end = node["end"]
            node_instances = instances[begin:end]
            label_priors = node["label_counts"]

            # Find the best split.
            best_gini = -1  # gini is always >= 0, so we can use -1 as "not set" value
            best_index = 0
            best_dim = 0
            split_dims = random.sample(dims, n_rand_dims)
            for d in split_dims:
                feats = data[node_instances, d]
                sorted_instances = numpy.argsort(feats)
                sorted_labels = labels[node_instances[sorted_instances]]

                gini_value, index, split_found = randomforest_functions.find_best_gini(sorted_labels, label_priors)

                if split_found:
                    if best_gini < 0:
                        best_gini = gini_value
                        best_index = index
                        best_dim = d
                    else:
                        if gini_value < best_gini:
                            best_gini = gini_value
                            best_index = index
                            best_dim = d

            assert best_gini > -1

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

            class_count_left = numpy.zeros((len(self._label_names),), dtype=numpy.int_)
            class_count_left[cl_left] = counts_left
            class_count_right = numpy.zeros((len(self._label_names),), dtype=numpy.int_)
            class_count_right[cl_right] = counts_right

            # Add the children to the graph.
            self._graph.add_node(next_node_id, begin=begin, end=middle,
                                 num_labels=len(cl_left), label_counts=class_count_left,
                                 is_left=True)
            self._graph.add_node(next_node_id+1, begin=middle, end=end,
                                 num_labels=len(cl_right), label_counts=class_count_right,
                                 is_left=False)
            self._graph.add_edge(node_id, next_node_id)
            self._graph.add_edge(node_id, next_node_id+1)

            # Update the node with the split information.
            node["split_dim"] = best_dim
            node["split_value"] = (data[instances[middle-1], best_dim] + data[instances[middle], best_dim]) / 2.

            if len(cl_left) > 1:
                qu.append(next_node_id)
            if len(cl_right) > 1:
                qu.append(next_node_id+1)
            next_node_id += 2

    def predict_proba(self, data):
        """
        Predict the class probabilities of the data.

        :param data: the data
        :return: class probabilities of the data
        """
        # Transform the graph information into arrays.
        num_nodes = self._graph.number_of_nodes()
        node_children = -numpy.ones((num_nodes, 2), numpy.int_)
        node_split_dims = numpy.zeros((num_nodes,), numpy.int_)
        node_split_values = numpy.zeros((num_nodes,), numpy.float_)
        node_label_count = numpy.zeros((num_nodes, len(self._label_names)), numpy.int_)
        for node_id in self._graph.nodes():
            node = self._graph.node[node_id]
            for j, c in enumerate(node["label_counts"]):
                node_label_count[node_id, j] = c
            if "split_dim" in node:
                n0_id, n1_id = self._graph.neighbors(node_id)
                if not self._graph.node[n0_id]["is_left"]:
                    assert self._graph.node[n1_id]["is_left"]
                    n0_id, n1_id = n1_id, n0_id
                node_children[node_id, 0] = n0_id
                node_children[node_id, 1] = n1_id
                node_split_dims[node_id] = node["split_dim"]
                node_split_values[node_id] = node["split_value"]

        # Call the cython probability function.
        probs = randomforest_functions.predict_proba(data.astype(numpy.float_), node_children, node_split_dims,
                                                     node_split_values, node_label_count)
        return probs

    def predict(self, data):
        """
        Predict classes of the data.

        :param data: the data
        :return: classes of the data
        """
        probs = self.predict_proba(data)
        pred = numpy.argmax(probs, axis=1)
        return self._label_names[pred]


def train_single_tree(tree, (data_ptr, data_dtype, data_shape), (labels_ptr, labels_dtype, labels_shape),
                      *args, **kwargs):
    """
    Train a single tree and return it.

    :param tree: the tree
    :param data_ptr: the c_ptr to the data array (can be obtained using arr.ctypes.data)
    :param data_dtype: the dtype of the data array
    :param data_shape: the shape of the data array
    :param labels_ptr: the c_ptr to the labels array (can be obtained using arr.ctypes.data)
    :param labels_dtype: the dtype of the labels array
    :param labels_shape: the shape of the labels array
    :return: the (trained) tree
    """
    # Create the numpy array from data_ptr.
    data_size = 1
    for s in data_shape:
        data_size *= s
    data_bfr = numpy.core.multiarray.int_asbuffer(data_ptr, data_size * data_dtype.itemsize)
    data = numpy.frombuffer(data_bfr, data_dtype).reshape(data_shape)

    # Create the labels array from labels_ptr.
    labels_size = 1
    for s in labels_shape:
        labels_size *= s
    labels_bfr = numpy.core.multiarray.int_asbuffer(labels_ptr, labels_size * labels_dtype.itemsize)
    labels = numpy.frombuffer(labels_bfr, labels_dtype).reshape(labels_shape)

    # Call the tree.fit function.
    tree.fit(data, labels, *args, **kwargs)
    return tree


class RandomForestClassifier(object):

    def __init__(self, n_estimators=10, n_rand_dims="auto", n_jobs=None):
        if n_rand_dims == "auto":
            tree_rand_dims = "auto_reduced"
        else:
            tree_rand_dims = "all"

        if n_jobs is None:
            n_jobs = multiprocessing.cpu_count()
        assert n_jobs > 0

        self._n_jobs = n_jobs
        self._trees = [DecisionTreeClassifier(n_rand_dims=tree_rand_dims) for _ in xrange(n_estimators)]
        self._label_names = None

    def fit(self, data, labels):
        """
        Train a random forest.

        :param data: the data
        :param labels: classes of the data
        """
        n_jobs = min(self._n_jobs, len(self._trees))
        if n_jobs == 1:
            for tree in self._trees:
                tree.fit(data, labels)
        else:
            data_info = (data.ctypes.data, data.dtype, data.shape)
            labels_info = (labels.ctypes.data, labels.dtype, labels.shape)
            with concurrent.futures.ProcessPoolExecutor(n_jobs) as executor:
                futures = []
                for i, tree in enumerate(self._trees):
                    futures.append((i, executor.submit(train_single_tree, tree, data_info, labels_info)))
                for i, future in futures:
                    self._trees[i] = future.result()

        self._label_names = self._trees[0].classes()
        for tree in self._trees[1:]:
            assert (self._label_names == tree.classes()).all()

    def predict_proba(self, data):
        """
        Predict the class probabilities of the data.

        :param data: the data
        :return: class probabilities of the data
        """
        probs = numpy.zeros((len(self._trees), data.shape[0], len(self._label_names)), dtype=numpy.float_)
        for i, tree in enumerate(self._trees):
            probs[i, :, :] = tree.predict_proba(data)
        return numpy.mean(probs, axis=0)

    def predict(self, data):
        """
        Predict classes of the data.

        :param data: the data
        :return: classes of the data
        """
        probs = self.predict_proba(data)
        pred = numpy.argmax(probs, axis=1)
        return self._label_names[pred]
