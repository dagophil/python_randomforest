import numpy
import networkx
import collections
import random
import randomforest_functions
import concurrent.futures
import multiprocessing
import bisect
import json
from timer import Timer
import ctypes
import platform
import scipy.sparse
from scipy.special import gammaln


class ScoreUpdater(object):

    def __init__(self, winner="lowest", **attrs):
        """
        Create the ScoreUpdater and initialize it with the given attributes.

        :param attrs: attribute dict
        """
        self._attrs = attrs
        self._best_score = None
        if winner == "lowest":
            self._check = self._check_lowest
        elif winner == "highest":
            self._check = self._check_highest
        else:
            raise Exception("ScoreUpdater: Unknown option winner=%s" % winner)

    def _check_lowest(self, score):
        return score < self._best_score

    def _check_highest(self, score):
        return score > self._best_score

    def update(self, score, **attrs):
        """
        If called for the first time: Save score and update the attribute dict.
        Else: if score < saved score, update the score and the attribute dict.

        :param score: the score value
        :param attrs: attribute dict
        """
        if self._best_score is None:
            self._best_score = score
            self._attrs.update(attrs)
        else:
            if self._check(score):
                self._best_score = score
                self._attrs.update(attrs)

    def updated(self):
        """
        Return True if the score was updated at least once.

        :return: whether the score was updated or not
        """
        return self._best_score is not None

    def __getitem__(self, item):
        """
        Return the value of the saved attribute.

        :param item: attribute name
        :return: attribute value
        """
        return self._attrs[item]


class DecisionTreeClassifier(object):

    def __init__(self, n_rand_dims="all", bootstrap_sampling=True, use_sample_label_count=True, resample_count=None,
                 max_depth=None, min_count=None, loggamma_tau=None, split_selection="gini"):
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
        :param loggamma_tau: tau value of the loggamma termination criterion
        :param split_selection: how to choose the split ("gini", "ksd")
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
        self._depth = 0
        assert loggamma_tau is None or (0 < loggamma_tau <= 1)
        self._loggamma_tau = loggamma_tau
        if split_selection == "gini":
            self._score_winner = "lowest"
            self._score_func = randomforest_functions.find_best_gini
        elif split_selection == "ksd":
            self._score_winner = "highest"
            self._score_func = randomforest_functions.find_best_ksd
        elif split_selection == "information_gain":
            self._score_winner = "lowest"
            self._score_func = randomforest_functions.find_best_ig
        else:
            raise Exception("Unknown split selection: %s" % split_selection)
        self._split_selection = split_selection

    def to_string(self):
        """
        Return a string representation of the decision tree classifier.

        :return: string
        """
        # Get the dict that can be given to the constructor.
        constructor_dict = {"n_rand_dims": self._n_rand_dims,
                            "bootstrap_sampling": self._bootstrap_sampling,
                            "use_sample_label_count": self._use_sample_label_count,
                            "resample_count": self._resample_count,
                            "max_depth": self._max_depth,
                            "min_count": self._min_count,
                            "loggamma_tau": self._loggamma_tau,
                            "split_selection": self._split_selection}

        # Get the graph information as numpy arrays.
        arrs = self._get_arrays()
        arrs_list = [(a.tolist(), a.dtype.str) for a in arrs]

        # Create the dict to be saved.
        d = {"constructor_dict": constructor_dict,
             "graph": arrs_list,
             "label_names": (self._label_names.tolist(), self._label_names.dtype.str),
             "depth": self._depth}

        return json.dumps(d)

    @staticmethod
    def from_string(s):
        """
        Create a decision tree classifier using the given string.

        :param s: string representation of decision tree
        :return: decision tree classifier
        """
        # Create the tree with the given options.
        d = json.loads(s)
        constructor_dict = d["constructor_dict"]
        tree = DecisionTreeClassifier(**constructor_dict)

        # Set the label names.
        label_names_list = d["label_names"][0]
        label_names_dtype = numpy.dtype(d["label_names"][1])
        tree._label_names = numpy.array(label_names_list, dtype=label_names_dtype)

        # Set the depth.
        tree._depth = d["depth"]

        # Create the graph.
        node_children, split_dims, split_values, label_count = [numpy.array(a[0], dtype=numpy.dtype(a[1]))
                                                                for a in d["graph"]]
        qu = collections.deque()
        qu.append((None, 0))
        tree._graph.add_node(0, depth=0)
        while len(qu) > 0:
            # Add the node to the graph.
            is_left, node_id = qu.popleft()
            node = tree._graph.node[node_id]
            node["label_counts"] = label_count[node_id, :]

            # Add split information.
            if split_dims[node_id] >= 0:
                node["split_dim"] = split_dims[node_id]
                node["split_value"] = split_values[node_id]

            # Add is_left information.
            if is_left is not None:
                node["is_left"] = is_left

            # Add the children to the graph and the queue.
            id_left = node_children[node_id, 0]
            if id_left >= 0:
                qu.append((True, id_left))
                tree._graph.add_node(id_left, depth=node["depth"]+1)
                tree._graph.add_edge(node_id, id_left)
            id_right = node_children[node_id, 1]
            if id_right >= 0:
                qu.append((False, id_right))
                tree._graph.add_node(id_right, depth=node["depth"]+1)
                tree._graph.add_edge(node_id, id_right)

        return tree

    def classes(self):
        """
        Return the classes that were found in training.

        :return: the classes
        """
        return numpy.array(self._label_names)

    def num_nodes(self):
        """
        Return the number of nodes in the graph.

        :return: number of nodes in the graph
        """
        return self._graph.number_of_nodes()

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
        node = self._graph.node[node_id]
        if "depth" in node:
            self._depth = max(self._depth, node["depth"])
            if self._max_depth is not None:
                if node["depth"] >= self._max_depth:
                    return True

        def check_counts(counts):
            """
            Check the following termination criteria on the count array:
            number of labels, min_count of instances, loggamma
            """
            s = numpy.sum(counts)
            if len(numpy.nonzero(counts)[0]) <= 1:
                return True
            if self._min_count is not None:
                if s <= self._min_count:
                    return True
            if self._loggamma_tau is not None:
                if numpy.sum(gammaln(counts+1)) - gammaln(s+1) >= numpy.log(self._loggamma_tau / 2.0):
                    return True
            return False

        if "label_counts" in node:
            if check_counts(node["label_counts"]):
                return True
        if "sample_label_counts" in node:
            if check_counts(node["sample_label_counts"]):
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
            score_updater = ScoreUpdater(winner=self._score_winner, index=0, dims=0)
            split_dims = random.sample(dims, n_rand_dims)
            for d in split_dims:
                feats = data[node_instances, d]
                sorted_instances = numpy.argsort(feats)
                sorted_labels = labels[node_instances[sorted_instances]]
                score, index, split_found = self._score_func(sorted_labels, label_priors)
                if split_found:
                    score_updater.update(score, index=index, dim=d)

            # Do not split if no split was found.
            if not score_updater.updated():
                continue
            best_index = score_updater["index"]
            best_dim = score_updater["dim"]

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
            score_updater = ScoreUpdater(winner=self._score_winner, index=0, dims=0)
            split_dims = random.sample(dims, n_rand_dims)
            for d in split_dims:
                feats = data[node_instances, d]
                sorted_instances = numpy.argsort(feats)
                sorted_labels = labels[node_instances[sorted_instances]]
                score, index, split_found = self._score_func(sorted_labels, label_priors)
                if split_found:
                    score_updater.update(score, index=index, dim=d)

            # Do not split if not split was found.
            if not score_updater.updated():
                continue
            best_index = score_updater["index"]
            best_dim = score_updater["dim"]

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
            score_updater = ScoreUpdater(winner=self._score_winner, index=0, dims=0)
            split_dims = random.sample(dims, n_rand_dims)
            for d in split_dims:
                feats = data[node_instances, d]
                sorted_instances = numpy.argsort(feats)
                sorted_labels = labels[node_instances[sorted_instances]]
                score, index, split_found = self._score_func(sorted_labels, label_priors)
                if split_found:
                    score_updater.update(score, index=index, dim=d)

            # Do not split if no split was found.
            if not score_updater.updated():
                continue
            best_index = score_updater["index"]
            best_dim = score_updater["dim"]

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

        if len(numpy.nonzero(numpy.isnan(data))[0]) > 0 or len(numpy.nonzero(numpy.isnan(labels))[0]) > 0:
            raise Exception("The input array contains NaNs")

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
                raise Exception("DecisionTreeClassifier.fit(): Unknown parameters.")
        else:
            self._fit_resample_count(data, labels)
            return

    def _get_arrays(self, return_node_sizes=False, return_num_leaves=False):
        """
        Return the children, the split dimensions, the split values and the label probabilities of each node as numpy arrays.

        :param return_node_sizes: if this is True, the number of instances in each node is returned
        :param return_num_leaves: if this is True, the number of leaves is returned
        :return: node_children, split_dimensions, split_values, label_probs, (optional: node_sizes), (optional: num_leaves)
        """
        num_nodes = self._graph.number_of_nodes()
        node_children = -numpy.ones((num_nodes, 2), numpy.int_)
        split_dims = -numpy.ones((num_nodes,), numpy.int_)
        split_values = numpy.zeros((num_nodes,), numpy.float_)
        label_probs = numpy.zeros((num_nodes, len(self._label_names)), numpy.float_)
        node_sizes = numpy.zeros((num_nodes,), numpy.int_)
        num_leaves = 0
        for node_id in self._graph.nodes():
            node = self._graph.node[node_id]

            # Update the label count.
            has_labels = False
            if "label_probs" in node:
                label_probs[node_id, :] = node["label_probs"]
                if node["label_probs"].sum() > 0:
                    has_labels = True
            else:
                assert "label_counts" in node
                node_sizes[node_id] = int(node["label_counts"].sum())
                for j, c in enumerate(node["label_counts"]):
                    label_probs[node_id, j] = c/float(node_sizes[node_id])
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
                split_dims[node_id] = node["split_dim"]
                split_values[node_id] = node["split_value"]
            else:
                num_leaves += 1

        ret = [node_children, split_dims, split_values, label_probs]
        if return_node_sizes:
            ret.append(node_sizes)
        if return_num_leaves:
            ret.append(num_leaves)
        return tuple(ret)

    def predict_proba(self, data, return_split_counts=False, node_weights=False):
        """
        Predict the class probabilities of the data.

        :param data: the data
        :param return_split_counts: if this is True, the number of comparisons is returned
        :param node_weights: if this is True, the probabilities are multiplied with the number of instances in the node
        :return: class probabilities of the data
        """
        if numpy.isnan(numpy.sum(data)):
            raise Exception("The input array contains NaNs")

        # Get the arrays with the node information.
        node_children, split_dims, split_values, label_probs, node_sizes = self._get_arrays(return_node_sizes=True)

        # Call the cython probability function.
        if return_split_counts:
            probs, counts = randomforest_functions.predict_proba(data.astype(numpy.float_), node_children, split_dims,
                                                                 split_values, label_probs, node_sizes,
                                                                 return_split_counts=True, node_weights=node_weights)
            return numpy.array(probs), counts
        else:
            probs = randomforest_functions.predict_proba(data.astype(numpy.float_), node_children, split_dims,
                                                         split_values, label_probs, node_sizes,
                                                         node_weights=node_weights)
            return numpy.array(probs)

    def predict(self, data, node_weights=False):
        """
        Predict classes of the data.

        :param data: the data
        :param node_weights: if this is True, the probabilities are multiplied with the number of instances in the node
        :return: classes of the data
        """
        probs = self.predict_proba(data, node_weights=node_weights)
        pred = numpy.argmax(probs, axis=1)
        return self._label_names[pred]

    def leaf_ids(self, data):
        """
        Return the leaf id of each instance in data.

        :param data: the data
        :return: leaf ids of the data
        """
        # Get the arrays with the node information.
        node_children, split_dims, split_values, label_probs = self._get_arrays()
        indices = randomforest_functions.leaf_ids(data.astype(numpy.float_), node_children, split_dims, split_values)
        return indices

    def node_weights(self):
        """
        Return the 2-class node weights

        :return: node weights
        """
        node_children, split_dims, split_values, label_probs = self._get_arrays()
        return label_probs[:, 1]

    def adjusted_node_weights(self):
        """
        Return the adjusted 2-class node weights.

        :return: node weights
        """
        node_children, split_dims, split_values, label_probs = self._get_arrays()
        return randomforest_functions.adjusted_node_weights(node_children, label_probs)

    def node_index_vectors(self, data):
        """
        Return the node index vector of each instance in data.

        :param data: the data
        :return: node index vectors (shape data.shape[0] x num_nodes, value is 1 if instance is in node else 0)
        """
        # Get the arrays with the node information.
        node_children, split_dims, split_values, label_probs = self._get_arrays()
        if self._depth == 0:
            print "Warning: Tree depth is zero."
        indices = randomforest_functions.node_ids_sparse(data.astype(numpy.float_), node_children, split_dims,
                                                         split_values, self._depth)
        return indices

    def weighted_index_vectors(self, data):
        """
        Return the weighted node index vector of each instance in data.

        :param data: the data
        :return: weighted node index vector
        """
        # Get the arrays with the node information.
        node_children, split_dims, split_values, label_probs = self._get_arrays()
        if self._depth == 0:
            print "Warning: Tree depth is zero."
        weights = randomforest_functions.weighted_node_ids_sparse(data.astype(numpy.float_), node_children, split_dims,
                                                                  split_values, label_probs, self._depth)
        return weights

    def leaf_index_vectors(self, data):
        """
        Return the leaf index vector of each instance in data.

        :param data: the data
        :return: leaf index vectors (shape data.shape[0] x num_leaves, value is 1 if instance is in node else 0)
        """
        # Get the arrays with the node information.
        node_children, split_dims, split_values, label_probs = self._get_arrays()
        if self._depth == 0:
            print "Warning: Tree depth is zero."
        indices = randomforest_functions.leaf_ids(data.astype(numpy.float_), node_children, split_dims, split_values)
        m = scipy.sparse.lil_matrix((data.shape[0], self.num_nodes()), dtype=numpy.int_)
        m[numpy.array(range(data.shape[0])), indices] = 1
        return m

    def sub_fg_tree(self, nodes, weights, num_trees):
        """
        Return a decision tree, where only the nodes in the path to the given nodes remain.
        Use the weights to assign new label probabilities according to the forest garrote.

        :param nodes: list with node ids
        :param weights: weights for the given nodes
        :param num_trees: number of trees from the forest garrote
        :return: sub decision tree
        """
        dt = DecisionTreeClassifier(n_rand_dims=self._n_rand_dims, bootstrap_sampling=self._bootstrap_sampling,
                                    use_sample_label_count=self._use_sample_label_count,
                                    resample_count=self._resample_count, max_depth=self._max_depth,
                                    min_count=self._min_count)
        dt._label_names = numpy.array(self._label_names)

        # Walk the tree from the given nodes to the root node and mark all nodes on the path to be kept.
        if len(nodes) == 0:
            return None
        keep = numpy.zeros(self.num_nodes(), dtype=numpy.uint8)
        keep[nodes] = 1
        for n in nodes:
            parents = self._graph.predecessors(n)
            while len(parents) > 0:
                p = parents[0]
                keep[p] = 1
                for c in self._graph.successors(p):
                    keep[c] = 1
                parents = self._graph.predecessors(p)
        assert keep[0] == 1  # the root node has to be kept

        # Make the weights accessible by node id.
        node_weights = numpy.zeros(self.num_nodes(), dtype=numpy.float_)
        node_weights[nodes] = weights

        # Build the new graph.
        node_map = numpy.zeros(self.num_nodes(), dtype=numpy.int_)
        gr = networkx.DiGraph()
        qu = collections.deque()
        qu.append(0)
        next_id = 0
        depth = 0
        while len(qu) > 0:
            # Get the node from the old graph and add it to the new graph.
            node_id = qu.popleft()
            node = self._graph.node[node_id]
            gr.add_node(next_id, depth=node["depth"])
            new_node = gr.node[next_id]
            node_map[node_id] = next_id
            depth = max(depth, new_node["depth"])

            # Update the node information.
            has_children = any([keep[c] == 1 for c in self._graph.successors(node_id)])
            if has_children:
                new_node["split_dim"] = node["split_dim"]
                new_node["split_value"] = node["split_value"]
            if "is_left" in node:
                new_node["is_left"] = node["is_left"]

            # Compute the forest garrote weight.
            if "label_probs" in node:
                new_node["prob"] = node["label_probs"][1]
            else:
                new_node["prob"] = node["label_counts"][1] / float(node["label_counts"].sum())
            new_node["fg_prob"] = new_node["prob"]
            if node_id != 0:
                p_id = node_map[self._graph.predecessors(node_id)[0]]
                new_node["fg_prob"] -= gr.node[p_id]["prob"]
                # Add the edge from the parent.
                gr.add_edge(p_id, next_id)
            new_node["weight"] = node_weights[node_id] * new_node["fg_prob"]
            if node_id != 0:
                p_id = node_map[self._graph.predecessors(node_id)[0]]
                new_node["weight"] += gr.node[p_id]["weight"]

            # Compute the label probability.
            w = new_node["weight"] * num_trees
            new_node["label_probs"] = numpy.array([1-w, w])

            # Add the children to the queue.
            for c in self._graph.successors(node_id):
                if keep[c] == 1:
                    qu.append(c)
            next_id += 1

        dt._graph = gr
        dt._depth = depth
        return dt

    def sub_gr_tree(self, nodes, weights, scale):
        """
        Return a decision tree, where only the nodes in the path to the given nodes remain.
        Use the weights to assign new label probabilities according to the global refinement.

        :param nodes: list with node ids
        :param weights weights for the given nodes
        :param scale: additional scaling for the weights
        :return: sub decision tree
        """
        dt = DecisionTreeClassifier(n_rand_dims=self._n_rand_dims, bootstrap_sampling=self._bootstrap_sampling,
                                    use_sample_label_count=self._use_sample_label_count,
                                    resample_count=self._resample_count, max_depth=self._max_depth,
                                    min_count=self._min_count)
        dt._label_names = numpy.array(self._label_names)

        # Walk the tree from the given nodes to the root node and mark all nodes on the path to be kept.
        if len(nodes) == 0:
            return None
        keep = numpy.zeros(self.num_nodes(), dtype=numpy.uint8)
        keep[nodes] = 1
        for n in nodes:
            parents = self._graph.predecessors(n)
            while len(parents) > 0:
                p = parents[0]
                keep[p] = 1
                for c in self._graph.successors(p):
                    keep[c] = 1
                parents = self._graph.predecessors(p)
        assert keep[0] == 1  # the root node has to be kept

        # Make the weights accessible by node id.
        node_weights = -numpy.ones((self.num_nodes(),), dtype=numpy.float_)
        node_weights[nodes] = weights

        # Build the new graph.
        node_map = numpy.zeros(self.num_nodes(), dtype=numpy.int_)
        gr = networkx.DiGraph()
        qu = collections.deque()
        qu.append(0)
        next_id = 0
        depth = 0
        while len(qu) > 0:
            # Get the node from the old graph and add it to the new graph.
            node_id = qu.popleft()
            node = self._graph.node[node_id]
            gr.add_node(next_id, depth=node["depth"])
            new_node = gr.node[next_id]
            node_map[node_id] = next_id
            depth = max(depth, new_node["depth"])

            # Update the node information.
            has_children = any([keep[c] == 1 for c in self._graph.successors(node_id)])
            if has_children:
                new_node["split_dim"] = node["split_dim"]
                new_node["split_value"] = node["split_value"]
            if "is_left" in node:
                new_node["is_left"] = node["is_left"]
            if node_id != 0:
                p_id = node_map[self._graph.predecessors(node_id)[0]]
                gr.add_edge(p_id, next_id)

            # Compute the label probability.
            if node_weights[node_id] >= 0:
                p = (1+node_weights[node_id]) / 2.0
                new_node["label_probs"] = numpy.array([1-p, p])
            else:
                new_node["label_probs"] = numpy.array([0.5, 0.5])

            # Add the children to the queue.
            for c in self._graph.successors(node_id):
                if keep[c] == 1:
                    qu.append(c)
            next_id += 1

        dt._graph = gr
        dt._depth = depth
        return dt


def train_single_tree(tree_id, data_id, labels_id, *args, **kwargs):
    """
    Train the given tree and return it.

    :param tree_id: id of the tree
    :param data_id: id of the data
    :param labels_id: id of the labels
    :return: the (trained) tree
    """
    # Seed the numpy random number generator.
    # This is necessary when using multiprocessing, otherwise the processes get the same seed.
    r = random.randint(0, numpy.iinfo(numpy.uint32).max-1)
    numpy.random.seed([r, multiprocessing.current_process().pid])

    # Convert the pointers to objects.
    tree = ctypes.cast(tree_id, ctypes.py_object).value
    data = ctypes.cast(data_id, ctypes.py_object).value
    labels = ctypes.cast(labels_id, ctypes.py_object).value

    # Call the tree.fit function.
    tree.fit(data, labels, *args, **kwargs)
    return tree


def predict_proba_single_tree(tree_id, data_id, return_split_counts=False, node_weights=False):
    """
    Predict using the given tree and return the probabilities.

    :param tree_id: id of the tree
    :param data_id: id of the data
    :param return_split_counts: if this is True, the number of comparisons is returned
    :param node_weights: if this is True, the probabilities are multiplied with the number of instances in the node
    :return: class probabilities
    """
    tree = ctypes.cast(tree_id, ctypes.py_object).value
    data = ctypes.cast(data_id, ctypes.py_object).value
    return tree.predict_proba(data, return_split_counts=return_split_counts, node_weights=node_weights)


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

    def classes(self):
        """
        Return the classes that were found in training.

        :return: the classes
        """
        return numpy.array(self._label_names)

    def num_trees(self):
        """
        Return the number of trees.

        :return: number of trees
        """
        return len(self._trees)

    def num_nodes(self, i=None):
        """
        Return the number of nodes in the forest. If i is not None, return the number of nodes of the i-th tree.

        :return: number of nodes
        """
        if i is None:
            return sum([tree.num_nodes() for tree in self._trees])
        else:
            return self._trees[i].num_nodes()

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
            if platform.python_implementation() != "CPython":
                raise Exception("You have to use CPython.")
            data_id = id(data)
            labels_id = id(labels)
            with concurrent.futures.ProcessPoolExecutor(n_jobs) as executor:
                futures = [(i, executor.submit(train_single_tree, id(tree), data_id, labels_id))
                           for i, tree in enumerate(self._trees)]
                for i, future in futures:
                    self._trees[i] = future.result()

        self._label_names = self._trees[0].classes()
        for tree in self._trees[1:]:
            assert (self._label_names == tree.classes()).all()

    def predict_proba(self, data, return_split_counts=False, node_weights=False):
        """
        Predict the class probabilities of the data.

        :param data: the data
        :param return_split_counts: if this is True, the number of comparisons is returned
        :param node_weights: if this is True, the probabilities are multiplied with the number of instances in the node
        :return: class probabilities of the data
        """
        probs = numpy.zeros((len(self._trees), data.shape[0], len(self._label_names)), dtype=numpy.float_)
        n_jobs = min(self._n_jobs, len(self._trees))
        counts = 0
        if n_jobs == 1:
            for i, tree in enumerate(self._trees):
                if return_split_counts:
                    probs[i, :, :], c = tree.predict_proba(data, return_split_counts=True, node_weights=node_weights)
                    counts += c
                else:
                    probs[i, :, :] = tree.predict_proba(data, node_weights=node_weights)
        else:
            if platform.python_implementation() != "CPython":
                raise Exception("You have to use CPython.")
            data_id = id(data)
            with concurrent.futures.ProcessPoolExecutor(n_jobs) as executor:
                futures = [(i, executor.submit(predict_proba_single_tree, id(tree), data_id, return_split_counts, node_weights))
                           for i, tree in enumerate(self._trees)]
                for i, future in futures:
                    if return_split_counts:
                        probs[i, :, :], c = future.result()
                        counts += c
                    else:
                        probs[i, :, :] = future.result()

        if node_weights:
            res = numpy.sum(probs, axis=0)
        else:
            res = numpy.mean(probs, axis=0)

        if return_split_counts:
            return res, counts
        else:
            return res

    def predict(self, data, return_split_counts=False, node_weights=False):
        """
        Predict classes of the data.

        :param data: the data
        :param return_split_counts: if this is True, the number of comparisons is returned
        :param node_weights: if this is True, the probabilities are multiplied with the number of instances in the node
        :return: classes of the data
        """
        if return_split_counts:
            probs, counts = self.predict_proba(data, return_split_counts=True, node_weights=node_weights)
            pred = numpy.argmax(probs, axis=1)
            return self._label_names[pred], counts
        else:
            probs = self.predict_proba(data, node_weights=node_weights)
            pred = numpy.argmax(probs, axis=1)
            return self._label_names[pred]

    def node_weights(self):
        """
        Return the node weights.

        :return: node weights
        """
        return numpy.concatenate([tree.node_weights() for tree in self._trees])

    def adjusted_node_weights(self):
        """
        Return the adjusted node weights.

        :return: adjusted node weights
        """
        return numpy.concatenate([tree.adjusted_node_weights() for tree in self._trees])

    def node_index_vectors(self, data):
        """
        Return the node index vector of each instance in data.

        :param data: the data
        :return: node index vectors (shape data.shape[0] x num_nodes, value is 1 if instance is in node else 0)
        """
        return scipy.sparse.hstack([tree.node_index_vectors(data) for tree in self._trees])

    def weighted_index_vectors(self, data):
        """
        Return the weighted index vector of each instance in data.

        :param data: the data
        :return: weighted index vectors
        """
        return scipy.sparse.hstack([tree.weighted_index_vectors(data) for tree in self._trees])

    def leaf_index_vectors(self, data):
        """
        Return the leaf index vector of each instance in data.

        :param data: the data
        :return: leaf index vectors (shape data.shape[0] x num_leaves, value is 1 if instance is in leaf else 0)
        """
        return scipy.sparse.hstack([tree.leaf_index_vectors(data) for tree in self._trees])

    def to_string(self):
        """
        Return a string that contains all information of the random forest classifier.

        :return: string with random forest information
        """
        d = {"trees": [t.to_string() for t in self._trees],
             "n_jobs": self._n_jobs,
             "label_names": (self._label_names.tolist(), self._label_names.dtype.str)}
        return json.dumps(d)

    @staticmethod
    def from_string(s):
        """
        Create a random forest classifier using the given string.

        :param s: string with random forest information
        :return: random forest classifier
        """
        d = json.loads(s)
        trees = [DecisionTreeClassifier.from_string(s) for s in d["trees"]]
        rf = RandomForestClassifier(n_estimators=0, n_jobs=d["n_jobs"])
        rf._trees = trees
        rf._label_names = numpy.array(d["label_names"][0], dtype=numpy.dtype(d["label_names"][1]))
        return rf

    def _sub_forest(self, nodes, weights, scale, tree_funcs):
        """
        Return a random forest, where only the nodes in the path to the given nodes remain.
        Use the functions in tree_funcs to construct the trees.

        :param nodes: list with node ids
        :param weights: weights for the given nodes
        :param scale: scale the weights in each node by this factor
        :param tree_funcs: container with tree construction functions
        :return: sub random forest
        """
        # Map the forest ids to the tree ids.
        node_counts = numpy.cumsum([0]+[tree.num_nodes() for tree in self._trees[:-1]])
        tree_nodes = [[] for _ in self._trees]
        tree_weights = [[] for _ in self._trees]
        for n, w in zip(nodes, weights):
            tree = bisect.bisect_right(node_counts, n)-1
            tree_nodes[tree].append(n - node_counts[tree])
            tree_weights[tree].append(w)
        tree_nodes = [numpy.array(n) for n in tree_nodes]
        tree_weights = [numpy.array(w) for w in tree_weights]

        # Build the random forest by taking the sub trees.
        rf = RandomForestClassifier(n_jobs=self._n_jobs)
        rf._label_names = numpy.array(self._label_names)
        rf._trees = [fn(n, w, scale) for fn, n, w in zip(tree_funcs, tree_nodes, tree_weights)]
        rf._trees = [tree for tree in rf._trees if tree is not None]
        return rf

    def sub_fg_forest(self, nodes, weights, scale):
        """
        Return a random forest, where only the nodes in the path to the given nodes remain.
        Use the weights to assign new label probabilities according to the forest garrote.

        :param nodes: list with node ids
        :param weights: weights for the given nodes
        :param scale: scale the weights in each node by this factor
        :return: sub random forest
        """
        return self._sub_forest(nodes, weights, scale, [tree.sub_fg_tree for tree in self._trees])

    def sub_gr_forest(self, nodes, weights):
        """
        Return a random forest, where only the nodes in the path to the given nodes remain.
        Use the weights to assign new label probabilities according to the global refinement.

        :param nodes: list with node ids
        :param weights: weights for the given nodes
        :return: sub random forest
        """
        return self._sub_forest(nodes, weights, 1.0, [tree.sub_gr_tree for tree in self._trees])
