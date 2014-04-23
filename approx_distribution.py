from utils import memoize
import numpy as np

class Distribution(object):
    """
    Base class for a distribution we wish to sample from.
    We must be able to efficiently compute the log probability of a binary string.
    """

    def __init__(self):
        self._dim = 1

    @memoize
    def log_p(self, x):
        """
        Log probability of x
        """
        return -np.Inf

    @property
    def dim(self):
        """
        Dimensionality of the distribution (e.g. number of bits)
        """
        return self._dim


class ExponentialWeightsDistribution(Distribution):
    """
    Exponential weights distribution to weight points according to a sequence
    of queries.
    """

    def __init__(self, dim, queries, weights):
        """
        The distribution is defined by a set of queries q(x): X->[0,1]
        and a set of real valued weights for how strongly we should value
        the query's output.
        """
        self._dim = dim
        self.queries = queries

        assert isinstance(weights, np.ndarray)
        self.weights = weights

    # @memoize
    def log_p(self, x):
        """
        Pr(x) = \prod_j exp{-w_j*q_j(x)} = exp{-\sum_j w_j q_j(x)}
        log Pr(x) = -\sum_j w_j q_j(x)
        """
        qs = np.array([q.evaluate(x) for q in self.queries])
        return -np.dot(qs, self.weights)