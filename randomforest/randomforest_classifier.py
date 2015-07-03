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
        """
        Create the GiniUpdater and initialize it with the given attributes.

        :param attrs: attribute dict
        """
        self._attrs = attrs
        self._best_gini = None

    def update(self, gini, **attrs):
        """
        If called for the first time: Save gini and update the attribute dict.
        Else: if gini < saved gini, update the gini and the attribute dict.

        :param gini: the gini value
        :param attrs: attribute dict
        """
        if self._best_gini is None:
            self._best_gini = gini
            self._attrs.update(attrs)
        else:
            if gini < self._best_gini:
                self._best_gini = gini
                self._attrs.update(attrs)

    def updated(self):
        """
        Return True if the gini was updated at least once.

        :return: whether the gini was updated or not
        """
        return self._best_gini is not None

    def __getitem__(self, item):
        """
        Return the value of the saved attribute.

        :param item: attribute name
        :return: attribute value
        """
        return self._attrs[item]


class DecisionTreeClassifier(object):

    def __init__(self, n_rand_dims="all", bootstrap_sampling=True, use_sample_label_count=True, resample_count=None,
                 max_depth=None, min_count=None):
        """
        Create a DecisionTreeClassifier.

        :param n_rand_dims: how many random features are used in each split ("all": use all features,
            "auto_reduced": use sqrt(num_features), some integer: use the given number)
        :param bootstrap_sampling: use bootstrap sampling
        :param use_sample_label_count: use the label counts of instance sample instead of the true instances
        :param resample_count: if this is an integer, create a sample with this many instances in each node and compute
            the best split from there
        :param max_depth: maximum tree depth
        :param min_count: do not split a node if its number of instances is <= min_count
        """
        self._n_rand_dims = n_rand_dims
        self._graph = networkx.DiGraph()
        self._label_names = None
        self._bootstrap_sampling = bootstrap_sampling
        self._use_sample_label_count = use_sample_label_count
        self._resample_count = resample_count
        self._max_depth = max_depth
        self._min_count = min_count
        if (not bootstrap_sampling) and (resample_count is None):
            self._use_sample_label_count = False
        if resample_count is not None:
            self._use_sample_label_count = False
            self._bootstrap_sampling = False

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

    def _is_terminal(self, node_id):
        """
        Check the termination criteria on the given node.

        :param node_id: the node id
        :return: true if a termination criteria is met
        """
        n = self._graph.node[node_id]
        if self._max_depth is not None and "depth" in n:
            if n["depth"] >= self._max_depth:
                return True
        if "label_counts" in n:
            count = 0
            for c in n["label_counts"]:
                if c > 0:
                    count += 1
            if count <= 1:
                return True
        if "sample_label_counts" in n:
            count = 0
            for c in n["sample_label_counts"]:
                if c > 0:
                    count += 1
            if count <= 1:
                return True
        if self._min_count is not None and "label_counts" in n:
            count = 0
            for c in n["label_counts"]:
                count += c
            if count <= self._min_count:
                return True
        if self._min_count is not None and "sample_label_counts" in n:
            count = 0
            for c in n["sample_label_counts"]:
                count += c
            if count <= self._min_count:
                return True

        return False

    def _fit_resample_count(self, data, labels):
        """
        Fit the tree without bootstrap sampling and with resampling in each node.

        :param data: the data
        :param labels: the labels
        """
        assert self._resample_count is not None

        # Get the number of feature dimensions that are considered in each split.
        n_rand_dims = self._find_n_rand_dims(data.shape)
        dims = range(data.shape[1])

        # Translate the labels to 0, 1, 2, ...
        self._label_names, labels, label_counts = numpy.unique(labels, return_inverse=True, return_counts=True)
        assert len(self._label_names) > 1

        # Create the index vector.
        instances = numpy.array(xrange(0, data.shape[0]))

        # Add the root node to the graph and the queue.
        self._graph.add_node(0, begin=0, end=data.shape[0], label_counts=label_counts, depth=0)
        next_node_id = 1
        qu = collections.deque()
        qu.append(0)

        # Split each node.
        while len(qu) > 0:
            node_id = qu.popleft()
            node = self._graph.node[node_id]
            begin = node["begin"]
            end = node["end"]
            depth = node["depth"]

            if self._resample_count < end-begin:
                num_samples = min(self._resample_count, end-begin)
                sample_indices = numpy.random.random_integers(begin, end-1, num_samples)
                node_instances = instances[sample_indices]
                num_classes, label_priors = self._get_split_class_count(labels, node_instances)

                # Do not split if no split is possible with the current sample.
                if num_classes <= 1:
                    continue
            else:
                node_instances = instances[begin:end]
                label_priors = node["label_counts"]

            # Find the best split.
            gini_updater = GiniUpdater(index=0, dims=0)
            split_dims = random.sample(dims, n_rand_dims)
            for d in split_dims:
                feats = data[node_instances, d]
                sorted_instances = numpy.argsort(feats)
                sorted_labels = labels[node_instances[sorted_instances]]
                gini, index, split_found = randomforest_functions.find_best_gini(sorted_labels, label_priors)
                if split_found:
                    gini_updater.update(gini, index=index, dim=d)

            # Do not split if no split was found.
            if not gini_updater.updated():
                continue
            best_index = gini_updater["index"]
            best_dim = gini_updater["dim"]

            # Sort the node instances.
            feats = data[node_instances, best_dim]
            sorted_instances = numpy.argsort(feats)
            node_instances = node_instances[sorted_instances]

            # Update the current node with the split information.
            node["split_dim"] = best_dim
            node["split_value"] = (data[node_instances[best_index-1], best_dim] +
                                   data[node_instances[best_index], best_dim]) / 2.0

            # Sort the index vector.
            node_instances = instances[begin:end]
            feats = data[node_instances, best_dim]
            sorted_instances = numpy.argsort(feats)
            instances[begin:end] = node_instances[sorted_instances]
            middle = begin + bisect.bisect_right(feats[sorted_instances], node["split_value"])

            num_classes_left, class_count_left = self._get_split_class_count(labels, instances[begin:middle])
            num_classes_right, class_count_right = self._get_split_class_count(labels, instances[middle:end])

            # Do not split if the split is invalid.
            if num_classes_left == 0 or num_classes_right == 0:
                del node["split_dim"]
                del node["split_value"]
                continue

            # Add the children to the graph.
            self._graph.add_node(next_node_id, begin=begin, end=middle, label_counts=class_count_left, is_left=True,
                                 depth=depth+1)
            self._graph.add_node(next_node_id+1, begin=middle, end=end, label_counts=class_count_right, is_left=False,
                                 depth=depth+1)
            self._graph.add_edge(node_id, next_node_id)
            self._graph.add_edge(node_id, next_node_id+1)

            # Check the termination conditions.
            if not self._is_terminal(next_node_id):
                qu.append(next_node_id)
            if not self._is_terminal(next_node_id+1):
                qu.append(next_node_id+1)
            next_node_id += 2

    def _fit_use_sample_label_count(self, data, labels):
        """
        Fit the tree using bootstrap sampling without resampling in each node. Use the out of bags in the prediction.

        :param data: the data
        :param labels: the labels
        """
        # Get the number of feature dimensions that are considered in each split.
        n_rand_dims = self._find_n_rand_dims(data.shape)
        dims = range(data.shape[1])

        # Translate the labels to 0, 1, 2, ...
        self._label_names, labels, label_counts = numpy.unique(labels, return_inverse=True, return_counts=True)
        assert len(self._label_names) > 1

        # Create the index vectors.
        instances = numpy.array(xrange(data.shape[0]))
        sample_instances = numpy.random.random_integers(0, data.shape[0]-1, data.shape[0])
        _, sample_label_counts = numpy.unique(labels[sample_instances], return_counts=True)

        # Add the root node to the graph and the queue
        self._graph.add_node(0, sample_begin=0, sample_end=data.shape[0], sample_label_counts=sample_label_counts,
                             begin=0, end=data.shape[0], label_counts=label_counts, depth=0)
        next_node_id = 1
        qu = collections.deque()
        qu.append(0)

        # Split each node.
        while len(qu) > 0:
            node_id = qu.popleft()
            node = self._graph.node[node_id]
            sample_begin = node["sample_begin"]
            sample_end = node["sample_end"]
            node_instances = sample_instances[sample_begin:sample_end]
            label_priors = node["sample_label_counts"]
            depth = node["depth"]

            # Find the best split.
            gini_updater = GiniUpdater(index=0, dims=0)
            split_dims = random.sample(dims, n_rand_dims)
            for d in split_dims:
                feats = data[node_instances, d]
                sorted_instances = numpy.argsort(feats)
                sorted_labels = labels[node_instances[sorted_instances]]
                gini, index, split_found = randomforest_functions.find_best_gini(sorted_labels, label_priors)
                if split_found:
                    gini_updater.update(gini, index=index, dim=d)

            # Do not split if not split was found.
            if not gini_updater.updated():
                continue
            best_index = gini_updater["index"]
            best_dim = gini_updater["dim"]

            # Sort the sample index vector.
            feats = data[node_instances, best_dim]
            sorted_instances = numpy.argsort(feats)
            sample_instances[sample_begin:sample_end] = node_instances[sorted_instances]
            sample_middle = sample_begin + best_index

            # Update the current node with the split information.
            node["split_dim"] = best_dim
            node["split_value"] = (data[sample_instances[sample_middle-1], best_dim] +
                                   data[sample_instances[sample_middle], best_dim]) / 2.0

            # Get the label count including the out of bags.
            begin = node["begin"]
            end = node["end"]
            node_instances = instances[begin:end]
            feats = data[node_instances, best_dim]
            sorted_instances = numpy.argsort(feats)
            instances[begin:end] = node_instances[sorted_instances]
            middle = begin + bisect.bisect_right(feats[sorted_instances], node["split_value"])

            num_classes_left, class_count_left = self._get_split_class_count(labels, instances[begin:middle])
            num_classes_right, class_count_right = self._get_split_class_count(labels, instances[middle:end])

            # Do not split if there are not out of bags in the children.
            if num_classes_left == 0 or num_classes_right == 0:
                del node["split_dim"]
                del node["split_value"]
                continue

            # Get the label counts of the children.
            sample_num_classes_left, sample_class_count_left = \
                self._get_split_class_count(labels, sample_instances[sample_begin:sample_middle])
            sample_num_classes_right, sample_class_count_right = \
                self._get_split_class_count(labels, sample_instances[sample_middle:sample_end])

            # Add the children to the graph.
            self._graph.add_node(next_node_id, begin=begin, end=middle, label_counts=class_count_left,
                                 sample_begin=sample_begin, sample_end=sample_middle,
                                 sample_label_counts=sample_class_count_left, is_left=True, depth=depth+1)
            self._graph.add_node(next_node_id+1, begin=middle, end=end, label_counts=class_count_right,
                                 sample_begin=sample_middle, sample_end=sample_end,
                                 sample_label_counts=sample_class_count_right, is_left=False, depth=depth+1)
            self._graph.add_edge(node_id, next_node_id)
            self._graph.add_edge(node_id, next_node_id+1)

            # Check the termination conditions.
            if not self._is_terminal(next_node_id):
                qu.append(next_node_id)
            if not self._is_terminal(next_node_id+1):
                qu.append(next_node_id+1)
            next_node_id += 2

    def _fit(self, data, labels, bootstrap_sampling):
        """
        Fit the tree without resampling in each node.

        :param data: the data
        :param labels: the labels
        :param bootstrap_sampling: whether to use bootstrap sampling
        """
        if bootstrap_sampling:
            ind = numpy.random.random_integers(0, data.shape[0]-1, data.shape[0])
            data = data[ind]
            labels = labels[ind]

        # Get the number of feature dimensions that are considered in each split.
        n_rand_dims = self._find_n_rand_dims(data.shape)
        dims = range(data.shape[1])

        # Translate the labels to 0, 1, 2, ...
        self._label_names, labels, label_counts = numpy.unique(labels, return_inverse=True, return_counts=True)
        assert len(self._label_names) > 1

        # Create the index vector.
        instances = numpy.array(xrange(0, data.shape[0]))

        # Add the root node to the graph and the queue.
        self._graph.add_node(0, begin=0, end=data.shape[0], label_counts=label_counts, depth=0)
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
            depth = node["depth"]

            # Find the best split.
            gini_updater = GiniUpdater(index=0, dims=0)
            split_dims = random.sample(dims, n_rand_dims)
            for d in split_dims:
                feats = data[node_instances, d]
                sorted_instances = numpy.argsort(feats)
                sorted_labels = labels[node_instances[sorted_instances]]
                gini, index, split_found = randomforest_functions.find_best_gini(sorted_labels, label_priors)
                if split_found:
                    gini_updater.update(gini, index=index, dim=d)

            # Do not split if no split was found.
            if not gini_updater.updated():
                continue
            best_index = gini_updater["index"]
            best_dim = gini_updater["dim"]

            # Sort the index vector.
            feats = data[node_instances, best_dim]
            sorted_instances = numpy.argsort(feats)
            instances[begin:end] = node_instances[sorted_instances]
            middle = begin + best_index

            # Update the current node with the split information.
            node["split_dim"] = best_dim
            node["split_value"] = (data[instances[middle-1], best_dim] + data[instances[middle], best_dim]) / 2.0

            # Get the label count of the children.
            num_classes_left, class_count_left = self._get_split_class_count(labels, instances[begin:middle])
            num_classes_right, class_count_right = self._get_split_class_count(labels, instances[middle:end])

            # Add the children to the graph.
            self._graph.add_node(next_node_id, begin=begin, end=middle, label_counts=class_count_left, is_left=True,
                                 depth=depth+1)
            self._graph.add_node(next_node_id+1, begin=middle, end=end, label_counts=class_count_right, is_left=False,
                                 depth=depth+1)
            self._graph.add_edge(node_id, next_node_id)
            self._graph.add_edge(node_id, next_node_id+1)

            # Check the termination conditions.
            if not self._is_terminal(next_node_id):
                qu.append(next_node_id)
            if not self._is_terminal(next_node_id+1):
                qu.append(next_node_id+1)
            next_node_id += 2

    def fit(self, data, labels):
        """
        Train a decision tree.

        :param data: the data
        :param labels: classes of the data
        """
        assert data.shape[0] == labels.shape[0]

        if self._resample_count is None:
            if self._bootstrap_sampling and self._use_sample_label_count:
                self._fit(data, labels, True)
                return
            elif (not self._bootstrap_sampling) and (not self._use_sample_label_count):
                self._fit(data, labels, False)
                return
            elif self._bootstrap_sampling and (not self._use_sample_label_count):
                self._fit_use_sample_label_count(data, labels)
                return
            else:
                raise Exception("Unknown parameters (resample_count is None).")
        else:
            self._fit_resample_count(data, labels)
            return

        # # GENERIC CASE:
        #
        # # Get the number of feature dimensions that are considered in each split.
        # n_rand_dims = self._find_n_rand_dims(data.shape)
        # dims = range(data.shape[1])
        #
        # # Translate the labels to 0, 1, 2, ...
        # self._label_names, labels, label_counts = numpy.unique(labels, return_inverse=True, return_counts=True)
        # assert len(self._label_names) > 1
        #
        # # Create the index vector.
        # instances = numpy.array(xrange(0, data.shape[0]))
        # if self._bootstrap_sampling:
        #     sample_instances = numpy.random.random_integers(0, data.shape[0]-1, data.shape[0])
        #     label_names, label_counts = numpy.unique(labels[sample_instances], return_counts=True)
        #     if len(self._label_names) != len(label_names):
        #         raise Exception("The sampling step removed one class completely.")
        # else:
        #     sample_instances = numpy.array(instances)
        #
        # # Add the root node to the graph and the queue.
        # self._graph.add_node(0, begin=0, end=data.shape[0],
        #                      begin_sample=0, end_sample=data.shape[0],
        #                      sample_label_counts=label_counts)
        # next_node_id = 1
        # qu = collections.deque()
        # qu.append(0)
        #
        # # Split each node.
        # while len(qu) > 0:
        #     node_id = qu.popleft()
        #     node = self._graph.node[node_id]
        #     begin_sample = node["begin_sample"]
        #     end_sample = node["end_sample"]
        #     sample_node_instances = sample_instances[begin_sample:end_sample]
        #     sample_label_priors = node["sample_label_counts"]
        #
        #     # Find the best split.
        #     gini_updater = GiniUpdater(index=0, dim=0)
        #     split_dims = random.sample(dims, n_rand_dims)
        #     for d in split_dims:
        #         feats = data[sample_node_instances, d]
        #         sorted_instances = numpy.argsort(feats)
        #         sorted_labels = labels[sample_node_instances[sorted_instances]]
        #         gini, index, split_found = randomforest_functions.find_best_gini(sorted_labels, sample_label_priors)
        #         if split_found:
        #             gini_updater.update(gini, index=index, dim=d)
        #
        #     # Do not split if no split was found.
        #     if not gini_updater.updated():
        #         continue
        #
        #     best_index = gini_updater["index"]
        #     best_dim = gini_updater["dim"]
        #
        #     # Sort the sample vector of the current node, so it can be easily split into two children.
        #     feats = data[sample_node_instances, best_dim]
        #     sorted_instances = numpy.argsort(feats)
        #     sample_instances[begin_sample:end_sample] = sample_node_instances[sorted_instances]
        #
        #     # Get the label count of the sample of each child.
        #     middle_sample = begin_sample+best_index
        #     num_classes_left, class_count_left = \
        #         self._get_split_class_count(labels, sample_instances[begin_sample:middle_sample])
        #     num_classes_right, class_count_right = \
        #         self._get_split_class_count(labels, sample_instances[middle_sample:end_sample])
        #
        #     # Update the current node with the split information.
        #     node["split_dim"] = best_dim
        #     node["split_value"] = (data[sample_instances[middle_sample-1], best_dim] +
        #                            data[sample_instances[middle_sample], best_dim]) / 2.0
        #
        #     left_properties = dict(begin_sample=begin_sample, end_sample=middle_sample,
        #                            sample_label_counts=class_count_left, is_left=True)
        #     right_properties = dict(begin_sample=middle_sample, end_sample=end_sample,
        #                             sample_label_counts=class_count_right, is_left=False)
        #
        #     if not self._use_sample_label_count:
        #         # Sort the instance vector of the current node, so it can be easily split into two children.
        #         begin = node["begin"]
        #         end = node["end"]
        #         node_instances = instances[begin:end]
        #         feats = data[node_instances, best_dim]
        #         sorted_instances = numpy.argsort(feats)
        #         instances[begin:end] = node_instances[sorted_instances]
        #         middle = begin + bisect.bisect_right(feats[sorted_instances], node["split_value"])
        #
        #         left_properties["begin"] = begin
        #         left_properties["end"] = middle
        #         right_properties["begin"] = middle
        #         right_properties["end"] = end
        #
        #     # Add the children to the graph.
        #     self._graph.add_node(next_node_id, left_properties)
        #     self._graph.add_node(next_node_id+1, right_properties)
        #     self._graph.add_edge(node_id, next_node_id)
        #     self._graph.add_edge(node_id, next_node_id+1)
        #
        #     if num_classes_left > 1:
        #         qu.append(next_node_id)
        #     if num_classes_right > 1:
        #         qu.append(next_node_id+1)
        #     next_node_id += 2
        # print "created", self._graph.number_of_nodes(), "nodes"
        #
        # # Update the label counts in the leaf nodes.
        # if self._use_sample_label_count:
        #     for node_id in self._graph.nodes():
        #         node = self._graph.node[node_id]
        #         node["label_counts"] = node["sample_label_counts"]
        # else:
        #     for node_id in self._graph.nodes():
        #         node = self._graph.node[node_id]
        #         begin = node["begin"]
        #         end = node["end"]
        #         num_classes, class_counts = self._get_split_class_count(labels, instances[begin:end])
        #         node["label_counts"] = class_counts

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
    Train the given tree and return it.

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

    def __init__(self, n_estimators=10, n_rand_dims="auto", n_jobs=None, **tree_kwargs):
        """
        Create a random forest classifier.
        :param n_estimators: number of trees
        :param n_rand_dims: how many random features are used in each split ("auto": use sqrt(num_features),
            "all": use all features, some integer: use the given number)
        :param n_jobs: number of parallel jobs
        :param tree_kwargs: additional arguments for the decision tree classifier.
        """
        if n_rand_dims == "auto":
            tree_rand_dims = "auto_reduced"
        else:
            tree_rand_dims = n_rand_dims

        if n_jobs is None:
            n_jobs = multiprocessing.cpu_count()
        assert n_jobs > 0

        self._n_jobs = n_jobs
        self._trees = [DecisionTreeClassifier(n_rand_dims=tree_rand_dims, **tree_kwargs)
                       for _ in xrange(n_estimators)]
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
