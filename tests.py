import sys
import vigra
import numpy
import randomforest
import time


def load_data(labels=None):
    """
    Load the data sets.

    :param labels: list with the labels that should be used
    :return: train_x, train_y, test_x, test_y
    """
    # Load the data.
    train_x = numpy.array(vigra.readHDF5("/home/philip/data/ml-koethe/train.h5", "data").transpose())
    train_y = vigra.readHDF5("/home/philip/data/ml-koethe/train.h5", "labels")
    test_x = numpy.array(vigra.readHDF5("/home/philip/data/ml-koethe/test.h5", "data").transpose())
    test_y = vigra.readHDF5("/home/philip/data/ml-koethe/test.h5", "labels")

    # Reduce the data to the given labels.
    if labels is not None:
        train_indices = numpy.array([i for i, t in enumerate(train_y) if t in labels])
        train_x = train_x[train_indices]
        train_y = train_y[train_indices]
        test_indices = numpy.array([i for i, t in enumerate(test_y) if t in labels])
        test_x = test_x[test_indices]
        test_y = test_y[test_indices]

    return train_x, train_y, test_x, test_y


def train_rf():
    """
    Train a random forest and compute the accuracy on a test set.
    """
    train_x, train_y, test_x, test_y = load_data([3, 8])
    dtree = randomforest.DecisionTreeClassifier("auto_reduced")
    start = time.time()
    dtree.fit(train_x, train_y)
    end = time.time()
    print end-start
    pred_y = dtree.predict(test_x)
    count = sum([1 if a == b else 0 for a, b in zip(test_y, pred_y)])
    print "%d of %d correct (%.02f%%)" % (count, len(pred_y), count/float(len(pred_y)))


def main():
    train_rf()
    return 0


if __name__ == "__main__":
    status = main()
    sys.exit(status)
