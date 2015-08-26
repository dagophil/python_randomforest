import sys
import vigra
import numpy
import randomforest
import argparse
from randomforest.timer import Timer
import os
import platform
from randomforest.forestgarrote import forest_garrote


def load_neuro_data():
    """
    Load the neuro dataset.

    :return: train_x, train_y, test_x, test_y
    """
    # Load the data.
    train_x = vigra.readHDF5("/home/philip/data/neuro_rf_test_data/niko/train/ffeat_br_segid0.h5", "ffeat_br")
    train_y = numpy.array(vigra.readHDF5("/home/philip/data/neuro_rf_test_data/niko/train/gt_face_segid0.h5", "gt_face")[:, 0])
    test_x = vigra.readHDF5("/home/philip/data/neuro_rf_test_data/niko/test/ffeat_br_segid0.h5", "ffeat_br")
    test_y = numpy.array(vigra.readHDF5("/home/philip/data/neuro_rf_test_data/niko/test/gt_face_segid0.h5", "gt_face")[:, 0])
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
    train_x = numpy.array(vigra.readHDF5("data/train.h5", "data").transpose())
    train_y = vigra.readHDF5("data/train.h5", "labels")
    test_x = numpy.array(vigra.readHDF5("data/test.h5", "data").transpose())
    test_y = vigra.readHDF5("data/test.h5", "labels")

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

    return args


def main():
    """
    Call the functions according to the command line arguments.

    :return: always return 0
    """
    args = parse_command_line()

    if args.dtree:
        train_dt(args.predict, args.save, args.load, args.filename)
    if args.rf:
        train_rf(args.n_trees, args.n_jobs, predict=args.predict, save=args.save, load=args.load,
                 filename=args.filename, refine=args.refine, group_size=args.group_size)

    return 0


if __name__ == "__main__":
    # Call the main function so the global namespace is not cluttered.
    status = main()
    sys.exit(status)
