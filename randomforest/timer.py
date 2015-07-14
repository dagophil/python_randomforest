import time


class Timer(object):

    def __init__(self, msg=None):
        if msg is None:
            msg = "took %f seconds"
        self.msg = msg
        self._start = None
        self._stop = None

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop = time.time()
        print self.msg % (self._stop - self._start)

