import sys
import vigra
import numpy
import randomforest
import argparse
from randomforest.timer import Timer
import os
import platform
from randomforest.forestgarrote import forest_garrote
from randomforest.refinement import global_refinement
import sklearn.cross_validation
import json
import multiprocessing
import time
import matplotlib.pyplot as plt


class TestResult(object):

    @staticmethod
    def _split_line(s):
        return tuple(float(x) for x in s.split(" "))

    @staticmethod
    def _create_result(lines):
        d = dict()
        d["performance"], d["performance_std"] = TestResult._split_line(lines[0])
        d["train_time"], d["train_time_std"] = TestResult._split_line(lines[1])
        d["split_counts"], d["split_counts_std"] = TestResult._split_line(lines[2])
        d["num_nodes"], d["num_nodes_std"] = TestResult._split_line(lines[3])
        return d

    def __init__(self, logfile_lines):
        self.params = json.loads(logfile_lines[0])
        self.result = self._create_result(logfile_lines[1:5])
        self.fg_001_result = self._create_result(logfile_lines[6:10])
        self.fg_0003_result = self._create_result(logfile_lines[11:15])
        self.fg_0001_result = self._create_result(logfile_lines[16:20])


def load_very_small_neuro_data():
    """
    Load the 1000 neuro dataset.

    :return: data_x, data_y
    """
    data_x = vigra.readHDF5("data/neuro/neuro_1000_raw_gt.h5", "raw")
    data_y = vigra.readHDF5("data/neuro/neuro_1000_raw_gt.h5", "gt")

    # Remove NaN values.
    to_remove = numpy.where(numpy.isnan(data_x))
    data_x = numpy.delete(data_x, to_remove, axis=0)
    data_y = numpy.delete(data_y, to_remove)

    return data_x, data_y


def load_small_neuro_data():
    """
    Load the small neuro dataset.

    :return: data_x, data_y
    """
    data_x = vigra.readHDF5("data/neuro/train/ffeat_br_segid0.h5", "ffeat_br")
    data_y = numpy.array(vigra.readHDF5("data/neuro/train/gt_face_segid0.h5", "gt_face")[:, 0])
    assert data_x.shape[0] == data_y.shape[0]

    # Remove NaN values.
    to_remove = numpy.where(numpy.isnan(data_x))
    data_x = numpy.delete(data_x, to_remove, axis=0)
    data_y = numpy.delete(data_y, to_remove)

    return data_x, data_y


def load_large_neuro_data():
    """
    Load the large neuro dataset.

    :return: data_x, data_y
    """
    data_x = vigra.readHDF5("data/neuro/test/ffeat_br_segid0.h5", "ffeat_br")
    data_y = numpy.array(vigra.readHDF5("data/neuro/test/gt_face_segid0.h5", "gt_face")[:, 0])
    assert data_x.shape[0] == data_y.shape[0]

    # Remove NaN values.
    to_remove = numpy.where(numpy.isnan(data_x))
    data_x = numpy.delete(data_x, to_remove, axis=0)
    data_y = numpy.delete(data_y, to_remove)
    return data_x, data_y


def load_neuro_data():
    """
    Load the neuro dataset.

    :return: train_x, train_y, test_x, test_y
    """
    # Load the data.
    train_x = vigra.readHDF5("data/neuro/train/ffeat_br_segid0.h5", "ffeat_br")
    train_y = numpy.array(vigra.readHDF5("data/neuro/train/gt_face_segid0.h5", "gt_face")[:, 0])
    test_x = vigra.readHDF5("data/neuro/test/ffeat_br_segid0.h5", "ffeat_br")
    test_y = numpy.array(vigra.readHDF5("data/neuro/test/gt_face_segid0.h5", "gt_face")[:, 0])
    assert train_x.shape[0] == train_y.shape[0]
    assert test_x.shape[0] == test_y.shape[0]
    assert train_x.shape[1] == test_x.shape[1]

    # Remove NaN values.
    to_remove = numpy.where(numpy.isnan(train_x))
    train_x = numpy.delete(train_x, to_remove, axis=0)
    train_y = numpy.delete(train_y, to_remove)
    to_remove = numpy.where(numpy.isnan(test_x))
    test_x = numpy.delete(test_x, to_remove, axis=0)
    test_y = numpy.delete(test_y, to_remove)

    return train_x, train_y, test_x, test_y


def load_data(labels=None):
    """
    Load the data sets.

    :param labels: list with the labels that should be used
    :return: train_x, train_y, test_x, test_y
    """
    # Load the data.
    train_x = numpy.array(vigra.readHDF5("data/mnist/train.h5", "data").transpose())
    train_y = vigra.readHDF5("data/mnist/train.h5", "labels")
    test_x = numpy.array(vigra.readHDF5("data/mnist/test.h5", "data").transpose())
    test_y = vigra.readHDF5("data/mnist/test.h5", "labels")

    # Reduce the data to the given labels.
    if labels is not None:
        train_indices = numpy.array([i for i, t in enumerate(train_y) if t in labels])
        train_x = train_x[train_indices]
        train_y = train_y[train_indices]
        test_indices = numpy.array([i for i, t in enumerate(test_y) if t in labels])
        test_x = test_x[test_indices]
        test_y = test_y[test_indices]

    return train_x, train_y, test_x, test_y


def train_dt(predict=True, save=False, load=False, filename=None):
    """
    Train a single decision tree and compute the accuracy on a test set.

    :param predict: use the decision to predict on a test set
    :param save: save the decision tree to a file
    :param load: load the decision tree from a file
    :param filename: file name
    """
    train_x, train_y, test_x, test_y = load_data([3, 8])

    if load:
        assert os.path.isfile(filename)
        print "Loading decision tree from file %s." % filename
        with open(filename, "r") as f:
            dtree_str = f.read()
        dtree = randomforest.DecisionTreeClassifier.from_string(dtree_str)
    else:
        print "Training decision tree."
        dtree = randomforest.DecisionTreeClassifier(n_rand_dims="auto_reduced")
        with Timer("Training took %.03f seconds."):
            dtree.fit(train_x, train_y)

    if save:
        print "Saving decision tree to file %s." % filename
        with open(filename, "w") as f:
            f.write(dtree.to_string())

    if predict:
        print "Predicting on a test set."
        with Timer("Prediction took %.03f seconds."):
            pred = dtree.predict(test_x)
        count = sum([1 if a == b else 0 for a, b in zip(test_y, pred)])
        print "%d of %d correct (%.03f%%)" % (count, len(pred), (100.0*count)/len(pred))


# # Train sklearn random forest.
# import sklearn.ensemble
# rf = sklearn.ensemble.RandomForestClassifier(n_estimators=8, n_jobs=8)
# rf.fit(train_x, train_y)
# pred = rf.predict(test_x)
# count = sum([1 if a == b else 0 for a, b in zip(test_y, pred)])
# print "%d of %d correct (%.03f%%)" % (count, len(pred), (100.0*count)/len(pred))

# # Train vigra random forest.
# rf = vigra.learning.RandomForest(treeCount=8)
# train_yy = train_y.reshape((train_y.shape[0], 1))
# rf.learnRF(train_x, train_yy)
# pred = rf.predictLabels(test_x)
# count = sum([1 if a == b else 0 for a, b in zip(test_y, pred)])
# print "%d of %d correct (%.03f%%)" % (count, len(pred), (100.0*count)/len(pred))


def train_rf(n_trees, n_jobs, predict=True, save=False, load=False, filename=None, refine=False, group_size=None):
    """
    Train a random forest and compute the accuracy on a test set.

    :param n_trees: number of trees
    :param n_jobs: number of jobs
    :param predict: use the random forest to predict on a test set
    :param save: save the random forest to a file
    :param load: load the random forest from a file
    :param filename: file name
    """
    # train_x, train_y, test_x, test_y = load_data([3, 8])
    train_x, train_y, test_x, test_y = load_neuro_data()

    if load:
        assert os.path.isfile(filename)
        print "Loading random forest from file %s." % filename
        with open(filename, "r") as f:
            rf_str = f.read()
        rf = randomforest.RandomForestClassifier.from_string(rf_str)
        if n_jobs is not None:
            rf._n_jobs = n_jobs
    else:
        print "Training random forest with %d trees." % n_trees
        rf = randomforest.RandomForestClassifier(n_estimators=n_trees, n_rand_dims="auto", n_jobs=n_jobs,
                                                 # bootstrap_sampling=True, use_sample_label_count=True, resample_count=None,
                                                 # bootstrap_sampling=False, use_sample_label_count=False, resample_count=None,
                                                 bootstrap_sampling=True, use_sample_label_count=False, resample_count=None,
                                                 # bootstrap_sampling=False, use_sample_label_count=True, resample_count=None,  # does not make sense
                                                 # resample_count=20,
                                                 # loggamma_tau=1e-6,
                                                 split_selection="gini"
                                                 )
        with Timer("Training took %.03f seconds"):
            rf.fit(train_x, train_y)
        print "The random forest has %d nodes." % rf.num_nodes()

    if save and not load:
        print "Saving random forest to file %s." % filename
        with open(filename, "w") as f:
            f.write(rf.to_string())

    if predict:
        print "Predicting on a test set with the random forest."
        with Timer("Random forest prediction took %.03f seconds."):
            pred, split_counts = rf.predict(test_x, return_split_counts=True)
        split_counts /= float(len(pred))
        count = sum([1 if a == b else 0 for a, b in zip(test_y, pred)])
        print "%d of %d correct (%.03f%%), used %.02f splits per instance" % (count, len(pred), (100.0*count)/len(pred), split_counts)

    if refine:
        print "Refining the random forest using forest garrote."
        with Timer("Refining took %.03f seconds."):
            refined_rf = forest_garrote(rf, train_x, train_y, group_size=group_size)
            # refined_rf = global_refinement(rf, train_x, train_y)
        print "The refined forest has %d nodes." % refined_rf.num_nodes()

        if save:
            f0, f1 = os.path.split(filename)
            refined_filename = os.path.join(f0, "refined_" + f1)
            print "Saving refined random forest to file %s." % refined_filename
            with open(refined_filename, "w") as f:
                f.write(refined_rf.to_string())

        if predict:
            print "Predicting on a test set with the forest garrote."
            with Timer("Forest garrote prediction took %.03f seconds."):
                pred, split_counts = refined_rf.predict(test_x, return_split_counts=True)
            split_counts /= float(len(pred))
            count = sum([1 if a == b else 0 for a, b in zip(test_y, pred)])
            print "%d of %d correct (%.03f%%), used %.02f splits per instance" % (count, len(pred), (100.0*count)/len(pred), split_counts)



def parameter_test_worker(i, p, num_params, alpha, group_size, data_x, data_y):
    """
    Get parameter sets from a queue and find the random forest performance. Send them back through another queue.

    :param qu_gl: queue for global objects, such as the dataset
    :param qu_in: queue for the parameter sets
    :param qu_out: queue for the output
    """
    out_str = "# Parameter set %d of %d" % (i+1, num_params) + "\n" + json.dumps(p) + "\n\n"

    rf_split_counts = []
    rf_performance = []
    rf_num_nodes = []
    rf_train_time = []
    fg = {a: {"split_counts": [],
              "performance": [],
              "num_nodes": [],
              "train_time": []}
          for a in alpha}

    kf = sklearn.cross_validation.KFold(data_x.shape[0], n_folds=10)
    for kf_i, (train, test) in enumerate(kf):
        print "## kf %d of %d" % (kf_i+1, len(kf))
        train_x = data_x[train]
        train_y = data_y[train]
        test_x = data_x[test]
        test_y = data_y[test]

        # Train the rf and get the performance.
        rf = randomforest.RandomForestClassifier(n_rand_dims="auto", n_jobs=1, **p)
        start = time.time()
        rf.fit(train_x, train_y)
        end = time.time()
        pred, split_counts = rf.predict(test_x, return_split_counts=True)
        split_counts /= float(len(pred))
        count = sum(1 for a, b in zip(test_y, pred) if a == b)
        performance = count/float(len(pred))

        rf_split_counts.append(split_counts)
        rf_performance.append(performance)
        rf_num_nodes.append(rf.num_nodes())
        rf_train_time.append(end-start)

        # Train the forest garrote and get the performance.
        for a_i, a in enumerate(alpha):
            print "## forest garrote %d of %d" % (a_i+1, len(alpha))
            start = time.time()
            if rf.num_trees() <= group_size:
                refined_rf = forest_garrote(rf, train_x, train_y, group_size=None, alpha=a)
            else:
                refined_rf = forest_garrote(rf, train_x, train_y, group_size=group_size, alpha=a)
            end = time.time()
            pred, split_counts = refined_rf.predict(test_x, return_split_counts=True)
            split_counts /= float(len(pred))
            count = sum(1 for a, b in zip(test_y, pred) if a == b)
            performance = count/float(len(pred))

            fg[a]["split_counts"].append(split_counts)
            fg[a]["performance"].append(performance)
            fg[a]["num_nodes"].append(refined_rf.num_nodes())
            fg[a]["train_time"].append(end-start)

    # Create the output string.
    out_str += "# performance\n" + str(numpy.mean(rf_performance)) + " " + str(numpy.std(rf_performance)) + "\n"
    out_str += "# train_time\n" + str(numpy.mean(rf_train_time)) + " " + str(numpy.std(rf_train_time)) + "\n"
    out_str += "# split_counts\n" + str(numpy.mean(rf_split_counts)) + " " + str(numpy.std(rf_split_counts)) + "\n"
    out_str += "# num_nodes\n" + str(numpy.mean(rf_num_nodes)) + " " + str(numpy.std(rf_num_nodes)) + "\n\n"
    for a in alpha:
        out_str += "fg " + str(a) + "\n\n"
        out_str += "# performance\n" + str(numpy.mean(fg[a]["performance"])) + " " + str(numpy.std(fg[a]["performance"])) + "\n"
        out_str += "# train_time\n" + str(numpy.mean(fg[a]["train_time"])) + " " + str(numpy.std(fg[a]["train_time"])) + "\n"
        out_str += "# split_counts\n" + str(numpy.mean(fg[a]["split_counts"])) + " " + str(numpy.std(fg[a]["split_counts"])) + "\n"
        out_str += "# num_nodes\n" + str(numpy.mean(fg[a]["num_nodes"])) + " " + str(numpy.std(fg[a]["num_nodes"])) + "\n\n"

    print out_str[:-1]
    with open("LOGFILE.txt", "a") as f:
        f.write(out_str)


def parameter_tests(dataset=0, n_jobs=None):
    """
    Train the random forest with different parameters and compute the cross validated score.
    Model properties, such as number of nodes, tree depth, ..., are printed to the output.

    :param dataset: which dataset is used
    :param n_jobs: number of parallel jobs
    """
    if n_jobs is None:
        n_jobs = multiprocessing.cpu_count()
    assert n_jobs > 0

    if dataset == 0:
        data_x, data_y = load_small_neuro_data()
    elif dataset == 1:
        data_x, data_y = load_large_neuro_data()
    elif dataset == 2:
        data_x, data_y = load_very_small_neuro_data()
    else:
        raise Exception("Dataset id unknown: %d" % dataset)

    # Create the random forest parameters.
    n_estimators = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    split_selection = ["gini", "ksd", "information_gain"]
    sampling_versions = [dict(bootstrap_sampling=True, use_sample_label_count=True, resample_count=None),
                         dict(bootstrap_sampling=False, use_sample_label_count=False, resample_count=None),
                         dict(bootstrap_sampling=True, use_sample_label_count=False, resample_count=None)]
    resample_count = [8, 16, 32, 64, 128, 256, 512, 1024]
    rf_params = []
    for n in n_estimators:
        for s in split_selection:
            for smpl in sampling_versions:
                d = dict(n_estimators=n, split_selection=s)
                d.update(smpl)
                rf_params.append(d)
            for r in resample_count:
                d = dict(n_estimators=n, split_selection=s, resample_count=r)
                rf_params.append(d)

    # Create the forest garrote parameters.
    alpha = [0.001, 0.0003, 0.0001]
    group_size = 4

    # Send the parameter sets.
    for i, p in enumerate(rf_params):
        parameter_test_worker(i, p, len(rf_params), alpha, group_size, data_x, data_y)


def plot_parameter_results(result_list, method_name, x_key, y_key, e_key=None, title=None, legend=None, restrictions=None,
                           anti_restrictions=None):
    if restrictions is None:
        restrictions = {}
    if anti_restrictions is None:
        anti_restrictions = {}

    fig, ax = plt.subplots(1, 1)

    for i, results in enumerate(result_list):
        for k in restrictions:
            results = [x for x in results if k not in x.params or x.params[k] == restrictions[k]]
        for k in anti_restrictions:
            results = [x for x in results if k in x.params and x.params[k] != anti_restrictions[k]]

        x_vals = [x.params[x_key] for x in results]
        y_vals = [x.__getattribute__(method_name)[y_key] for x in results]

        if e_key is None:
            ax.plot(x_vals, y_vals, marker="o")
        if e_key is not None:
            x_vals = [x+0.1*i for x in x_vals]
            e_vals = [x.result[e_key] for x in results]
            ax.errorbar(x_vals, y_vals, e_vals, marker="o")

    if title is not None:
        ax.set_title(title)
    if legend is not None:
        ax.legend(legend, loc=4)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.grid()
    plt.show()


def analyze_parameters(filename):
    """
    Get the results from the performance tests.

    :param filename: filename
    """
    with open(filename) as f:
        lines = [line.strip() for line in f if len(line) > 1 and line[0] != "#"]
    line_blocks = [lines[20*i:20*(i+1)] for i in xrange(len(lines)/20)]
    results = [TestResult(block) for block in line_blocks]

    gini_results = [x for x in results if x.params["split_selection"] == "gini"]
    ksd_results = [x for x in results if x.params["split_selection"] == "ksd"]
    ig_results = [x for x in results if x.params["split_selection"] == "information_gain"]

    # # performance vs num trees
    # plot_parameter_results([gini_results, ksd_results, ig_results], "result", x_key="n_estimators", y_key="performance",
    #                        e_key="performance_std", legend=["gini", "ksd", "ig"], title="random forest performance",
    #                        restrictions=dict(use_sample_label_count=True, bootstrap_sampling=True, resample_count=None))
    #
    # # split counts vs num trees
    # plot_parameter_results([gini_results, ksd_results, ig_results], "result", x_key="n_estimators", y_key="split_counts",
    #                        e_key="split_counts_std", legend=["gini", "ksd", "ig"], restrictions=dict(use_sample_label_count=True,
    #                        bootstrap_sampling=True, resample_count=None), title="random forest split counts")
    #
    # # forest garrote performance vs num trees
    # plot_parameter_results([gini_results, ksd_results, ig_results], "fg_0003_result", x_key="n_estimators", y_key="performance",
    #                        e_key="performance_std", legend=["gini", "ksd", "ig"], restrictions=dict(use_sample_label_count=True,
    #                        bootstrap_sampling=True, resample_count=None), title="forest garrote performance")
    #
    # # forest garrote split counts vs num trees
    # plot_parameter_results([gini_results, ksd_results, ig_results], "fg_0003_result", x_key="n_estimators", y_key="split_counts",
    #                        e_key="split_counts_std", legend=["gini", "ksd", "ig"], restrictions=dict(use_sample_label_count=True,
    #                        bootstrap_sampling=True, resample_count=None), title="forest garrote split counts")
    #
    # # performance vs resample count
    # plot_parameter_results([gini_results, ksd_results, ig_results], "result", x_key="resample_count", y_key="performance",
    #                        e_key="performance_std", legend=["gini", "ksd", "ig"], title="random forest performance",
    #                        restrictions=dict(n_estimators=16),
    #                        anti_restrictions=dict(resample_count=None))
    #
    # # train time vs resample count
    # plot_parameter_results([gini_results, ksd_results, ig_results], "result", x_key="resample_count", y_key="train_time",
    #                        e_key="train_time_std", legend=["gini", "ksd", "ig"], title="random forest training time",
    #                        restrictions=dict(n_estimators=16),
    #                        anti_restrictions=dict(resample_count=None))


def parse_command_line():
    """
    Parse the command line arguments.

    :return: command line arguments
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="A python random forest implementation.")
    parser.add_argument("--dtree", action="store_true", help="train a single decision tree")
    parser.add_argument("--rf", action="store_true", help="train a random forest")
    parser.add_argument("--predict", action="store_true", help="use the classifier to predict on a test set")
    parser.add_argument("--save", action="store_true", help="save the classifier to a file")
    parser.add_argument("--load", action="store_true", help="load the classifier from a file")
    parser.add_argument("--filename", type=str, help="file name")
    parser.add_argument("-n", "--n_trees", type=int, default=100, help="number of trees in the random forest")
    parser.add_argument("--n_jobs", type=int, default=-1, help="number of jobs (-1: use number of cores)")
    parser.add_argument("--refine", action="store_true", help="do the forest garrote refinement")
    parser.add_argument("--group_size", type=int, default=None, help="group size for the forest garrote")
    parser.add_argument("--parameter_tests", action="store_true", help="run performance tests on different parameter sets")
    parser.add_argument("--dataset", type=int, default=0, help="which dataset is used for the performance tests")
    parser.add_argument("--analyze", type=str, default=None, help="result file from the performance tests")
    args = parser.parse_args()

    if not args.dtree and not args.rf:
        args.rf = True
    if args.n_jobs <= 0:
        args.n_jobs = None
    if args.n_jobs is None and platform.python_implementation() != "CPython":
        raise Exception("It seems that the current interpreter does not use CPython. This is a problem, since the "
                        "random forest parallelization currently relies on a CPython implementation detail. Let me "
                        "know, if this is a problem for you.")
    if args.save or args.load:
        assert args.filename is not None
    if args.parameter_tests:
        print "# Running parameter tests. Only the arguments --n_jobs and --dataset are used."
    elif args.analyze is not None:
        assert os.path.isfile(args.analyze)
        print "# Analyzing file %s. No other arguments are used." % args.analyze

    return args


def main():
    """
    Call the functions according to the command line arguments.

    :return: always return 0
    """
    args = parse_command_line()

    if args.parameter_tests:
        parameter_tests(dataset=args.dataset, n_jobs=args.n_jobs)
        return

    if args.analyze is not None:
        analyze_parameters(args.analyze)
        return

    if args.dtree:
        train_dt(args.predict, args.save, args.load, args.filename)

    if args.rf:
        train_rf(args.n_trees, args.n_jobs, predict=args.predict, save=args.save, load=args.load,
                 filename=args.filename, refine=args.refine, group_size=args.group_size)


if __name__ == "__main__":
    # Call the main function so the global namespace is not cluttered.
    main()
    sys.exit(0)
