import numpy
import networkx
import collections
import random
import randomforest_functions
import time
import concurrent.futures
import multiprocessing
import bisect


class GiniUpdater(object):

    def __init__(self, **attrs):
        self._attrs = attrs
        self._best_gini = None

    def update(self, gini, **attrs):
        if self._best_gini is None:
            self._best_gini = gini
            self._attrs.update(attrs)
        else:
            if gini < self._best_gini:
                self._best_gini = gini
                self._attrs.update(attrs)

    def updated(self):
        return self._best_gini is not None

    def __getitem__(self, item):
        return self._attrs[item]


class DecisionTreeClassifier(object):

    def __init__(self, n_rand_dims="all", bootstrap_sampling=False, use_sample_label_count=True):
        self._n_rand_dims = n_rand_dims
        self._graph = networkx.DiGraph()
        self._label_names = None
        self._bootstrap_sampling = bootstrap_sampling
        self._use_sample_label_count = use_sample_label_count

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

    def _get_split_class_count(self, labels, instances):
        """
        The the number of classes and the class count of the given instances.

        :param labels: the labels
        :param instances: the instance vector
        :return: number of classes, class count
        """
        classes, counts = numpy.unique(labels[instances], return_counts=True)
        class_count = numpy.zeros((len(self._label_names),), dtype=numpy.int_)
        class_count[classes] = counts
        return len(classes), class_count

    def fit(self, data, labels):
        """
        Train a decision tree.

        :param data: the data
        :param labels: classes of the data
        """
        assert data.shape[0] == labels.shape[0]

        # Get the number of feature dimensions that are considered in each split.
        n_rand_dims = self._find_n_rand_dims(data.shape)
        dims = range(data.shape[1])

        # Translate the labels to 0, 1, 2, ...
        self._label_names, labels, label_counts = numpy.unique(labels, return_inverse=True, return_counts=True)
        assert len(self._label_names) > 1

        # Create the index vector.
        instances = numpy.array(xrange(0, data.shape[0]))
        if self._bootstrap_sampling:
            sample_instances = numpy.random.random_integers(0, data.shape[0]-1, data.shape[0])
            label_names, label_counts = numpy.unique(labels[sample_instances], return_counts=True)
            if len(self._label_names) != len(label_names):
                raise Exception("The sampling step removed one class completely.")
        else:
            sample_instances = numpy.array(instances)

        # Add the root node to the graph and the queue.
        self._graph.add_node(0, begin=0, end=data.shape[0],
                             begin_sample=0, end_sample=data.shape[0],
                             sample_label_counts=label_counts)
        next_node_id = 1
        qu = collections.deque()
        qu.append(0)

        # Split each node.
        while len(qu) > 0:
            node_id = qu.popleft()
            node = self._graph.node[node_id]

            # TODO: Resample eventually.
            begin_sample = node["begin_sample"]
            end_sample = node["end_sample"]

            sample_node_instances = sample_instances[begin_sample:end_sample]
            sample_label_priors = node["sample_label_counts"]

            # Find the best split.
            gini_updater = GiniUpdater(index=0, dim=0)
            split_dims = random.sample(dims, n_rand_dims)
            for d in split_dims:
                feats = data[sample_node_instances, d]
                sorted_instances = numpy.argsort(feats)
                sorted_labels = labels[sample_node_instances[sorted_instances]]
                gini, index, split_found = randomforest_functions.find_best_gini(sorted_labels, sample_label_priors)
                if split_found:
                    gini_updater.update(gini, index=index, dim=d)

            # Do not split if no split was found.
            if not gini_updater.updated():
                continue

            best_index = gini_updater["index"]
            best_dim = gini_updater["dim"]

            # TODO: Do not do this when resampling in each node.
            # Sort the sample vector of the current node, so it can be easily split into two children.
            feats = data[sample_node_instances, best_dim]
            sorted_instances = numpy.argsort(feats)
            sample_instances[begin_sample:end_sample] = sample_node_instances[sorted_instances]

            # Get the label count of the sample of each child.
            middle_sample = begin_sample+best_index
            num_classes_left, class_count_left = \
                self._get_split_class_count(labels, sample_instances[begin_sample:middle_sample])
            num_classes_right, class_count_right = \
                self._get_split_class_count(labels, sample_instances[middle_sample:end_sample])

            # Update the current node with the split information.
            node["split_dim"] = best_dim
            node["split_value"] = (data[sample_instances[middle_sample-1], best_dim] +
                                   data[sample_instances[middle_sample], best_dim]) / 2.0

            left_properties = dict(begin_sample=begin_sample, end_sample=middle_sample,
                                   sample_label_counts=class_count_left, is_left=True)
            right_properties = dict(begin_sample=middle_sample, end_sample=end_sample,
                                    sample_label_counts=class_count_right, is_left=False)

            if not self._use_sample_label_count:
                # Sort the instance vector of the current node, so it can be easily split into two children.
                begin = node["begin"]
                end = node["end"]
                node_instances = instances[begin:end]
                feats = data[node_instances, best_dim]
                sorted_instances = numpy.argsort(feats)
                instances[begin:end] = node_instances[sorted_instances]
                middle = begin + bisect.bisect_right(feats[sorted_instances], node["split_value"])

                left_properties["begin"] = begin
                left_properties["end"] = middle
                right_properties["begin"] = middle
                right_properties["end"] = end

            # Add the children to the graph.
            self._graph.add_node(next_node_id, left_properties)
            self._graph.add_node(next_node_id+1, right_properties)
            self._graph.add_edge(node_id, next_node_id)
            self._graph.add_edge(node_id, next_node_id+1)

            if num_classes_left > 1:
                qu.append(next_node_id)
            if num_classes_right > 1:
                qu.append(next_node_id+1)
            next_node_id += 2

        # Update the label counts in the leaf nodes.
        if self._use_sample_label_count:
            for node_id in self._graph.nodes():
                node = self._graph.node[node_id]
                node["label_counts"] = node["sample_label_counts"]
        else:
            for node_id in self._graph.nodes():
                node = self._graph.node[node_id]
                begin = node["begin"]
                end = node["end"]
                num_classes, class_counts = self._get_split_class_count(labels, instances[begin:end])
                node["label_counts"] = class_counts

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

            # Update the label count.
            has_labels = False
            for j, c in enumerate(node["label_counts"]):
                node_label_count[node_id, j] = c
                if c > 0:
                    has_labels = True

            # Update the children and split information.
            if "split_dim" in node and has_labels:
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
