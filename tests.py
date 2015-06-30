import sys
import vigra
import numpy
import randomforest
import argparse
import time


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
    rf = randomforest.RandomForestClassifier(n_estimators=n_trees, n_rand_dims="auto", n_jobs=n_jobs)

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
