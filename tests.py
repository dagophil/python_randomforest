import sys
import vigra
import numpy
import randomforest
import argparse
import time


def load_neuro_data():
    """
    Load the neuro dataset.

    :return: train_x, train_y, test_x, test_y
    """
    # Load the data.
    train_x = vigra.readHDF5("/home/philip/data/neuro_rf_test_data/train/ffeat_br_segid0.h5", "ffeat_br")
    train_y = numpy.array(vigra.readHDF5("/home/philip/data/neuro_rf_test_data/train/gt_face_segid0.h5", "gt_face")[:, 0])
    test_x = vigra.readHDF5("/home/philip/data/neuro_rf_test_data/test/ffeat_br_segid0.h5", "ffeat_br")
    test_y = numpy.array(vigra.readHDF5("/home/philip/data/neuro_rf_test_data/test/gt_face_segid0.h5", "gt_face")[:, 0])
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


def train_dt():
    """
    Train a single decision tree and compute the accuracy on a test set.
    """
    print "Train a single decision tree."
    train_x, train_y, test_x, test_y = load_data([3, 8])
    dtree = randomforest.DecisionTreeClassifier(n_rand_dims="auto_reduced")

    start = time.time()
    dtree.fit(train_x, train_y)
    end = time.time()
    print "Training took %.03f seconds." % (end-start)

    start = time.time()
    pred = dtree.predict(test_x)
    end = time.time()
    print "Prediction took %.03f seconds." % (end-start)

    count = sum([1 if a == b else 0 for a, b in zip(test_y, pred)])
    print "%d of %d correct (%.03f%%)" % (count, len(pred), (100.0*count)/len(pred))


def train_rf(n_trees, n_jobs):
    """
    Train a random forest and compute the accuracy on a test set.

    :param n_trees: number of trees
    :param n_jobs: number of jobs
    """
    print "Train a random forest with %d trees." % n_trees

    train_x, train_y, test_x, test_y = load_data([3, 8])
    # train_x, train_y, test_x, test_y = load_neuro_data()

    # import sklearn.ensemble
    # rf = sklearn.ensemble.RandomForestClassifier(n_estimators=8, n_jobs=8)
    # rf.fit(train_x, train_y)
    # pred = rf.predict(test_x)
    # count = sum([1 if a == b else 0 for a, b in zip(test_y, pred)])
    # print "%d of %d correct (%.03f%%)" % (count, len(pred), (100.0*count)/len(pred))
    #
    # rf = vigra.learning.RandomForest(treeCount=8)
    # train_yy = train_y.reshape((train_y.shape[0], 1))
    # rf.learnRF(train_x, train_yy)
    # pred = rf.predictLabels(test_x)
    # count = sum([1 if a == b else 0 for a, b in zip(test_y, pred)])
    # print "%d of %d correct (%.03f%%)" % (count, len(pred), (100.0*count)/len(pred))

    rf = randomforest.RandomForestClassifier(n_estimators=n_trees, n_rand_dims="auto", n_jobs=n_jobs,
                                             bootstrap_sampling=True, use_sample_label_count=True, resample_count=None,
                                             # bootstrap_sampling=False, use_sample_label_count=False, resample_count=None,
                                             # bootstrap_sampling=True, use_sample_label_count=False, resample_count=None,
                                             # bootstrap_sampling=False, use_sample_label_count=True, resample_count=None,  # does not make sense
                                             # resample_count=20,
                                             )

    start = time.time()
    rf.fit(train_x, train_y)
    end = time.time()
    print "Training took %.03f seconds." % (end-start)

    start = time.time()
    pred = rf.predict(test_x)
    end = time.time()
    print "Prediction took %.03f seconds." % (end-start)

    count = sum([1 if a == b else 0 for a, b in zip(test_y, pred)])
    print "%d of %d correct (%.03f%%)" % (count, len(pred), (100.0*count)/len(pred))


def parse_command_line():
    """
    Parse the command line arguments.

    :return: command line arguments
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="A python random forest implementation.")
    parser.add_argument("--dtree", action="store_true", help="train a single decision tree")
    parser.add_argument("--rf", action="store_true", help="train a random forest")
    parser.add_argument("-n", "--n_trees", type=int, default=100, help="number of trees in the random forest")
    parser.add_argument("--n_jobs", type=int, default=-1, help="number of jobs (-1: use number of cores)")
    args = parser.parse_args()

    if not args.dtree and not args.rf:
        args.rf = True
    if args.n_jobs <= 0:
        args.n_jobs = None

    return args


def main():
    """
    Call the functions according to the command line arguments.

    :return: always return 0
    """
    args = parse_command_line()

    if args.dtree:
        train_dt()
    if args.rf:
        train_rf(args.n_trees, args.n_jobs)

    return 0


if __name__ == "__main__":
    # Call the main function so the global namespace is not cluttered.
    status = main()
    sys.exit(status)
