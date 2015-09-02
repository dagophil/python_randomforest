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

    def worker(qu_in, qu_out):
        while True:
            item = qu_in.get()
            if item is None:
                break
            i, p = item

            out_str = "# Parameter set %d of %d" % (i+1, len(rf_params)) + "\n" + json.dumps(p) + "\n\n"

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
            for train, test in kf:
                train_x = data_x[train]
                train_y = data_y[train]
                test_x = data_x[test]
                test_y = data_y[test]

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

                for a in alpha:
                    start = time.time()
                    if rf.num_trees() < group_size:
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

            qu_out.put((i, out_str))

    # Create the workers and fill the queue.
    qu_input = multiprocessing.Queue()
    qu_output = multiprocessing.Queue()
    procs = [multiprocessing.Process(target=worker, args=(qu_input, qu_output)) for _ in xrange(n_jobs)]
    for i, p in enumerate(rf_params):
        qu_input.put((i, p))

    # Start the workers.
    for p in procs:
        p.start()
        qu_input.put(None)

    # Collect the worker output and print it.
    for _ in rf_params:
        i, s = qu_output.get()
        print s[:-1]

    # Wait for the processes.
    for p in procs:
        p.join()


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

    if args.dtree:
        train_dt(args.predict, args.save, args.load, args.filename)

    if args.rf:
        train_rf(args.n_trees, args.n_jobs, predict=args.predict, save=args.save, load=args.load,
                 filename=args.filename, refine=args.refine, group_size=args.group_size)


if __name__ == "__main__":
    # Call the main function so the global namespace is not cluttered.
    main()
    sys.exit(0)
