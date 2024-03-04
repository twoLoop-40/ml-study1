import numpy as np
from sklearn import datasets, metrics


class Digit:
    def __init__(self):
        self._source_digits = datasets.load_digits()
        self._digits = None

    def transform_(self):
        n_samples = len(self._source_digits.images)
        reshped = self._source_digits.images.reshape((n_samples, -1))
        return reshped

    @property
    def digits(self):
        if self._digits is None:
            self._digits = self.transform_()
        return self._digits
    
    @property
    def target(self):
        return self._source_digits.target

    

