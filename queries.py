"""
Define a set of useful queries
"""
import numpy as np

def sample_random_query(qtype, *args):
    # TODO: Sample a random query of the specified type
    return None

class Query(object):
    def __init__(self):
        pass

    def evaluate(self, x):
        """
        Evaluate the query on a given input
        """
        return 0.0


class Conjunction(Query):
    """
    A conjunction query q(x) = \wedge_j x_j
    """
    def __init__(self, dim, literals):
        """
        Initialize with a set of literals (bits) in the conjunction
        Literals can either be a binary bit string representing the
        conjunction, or a list of indices denoting literals in the
        conjunction.
        """
        self.dim = dim
        if isinstance(literals, np.ndarray) and \
                        literals.size == dim and \
                        np.amax(literals) <= 1 and \
                        np.amin(literals) >= 0:
            self.literals = literals
        elif isinstance(literals, np.ndarray) and \
                        literals.dtype == np.int and \
                        np.amax(literals) < self.dim and \
                        np.amin(literals) >= 0:
            self.literals = np.zeros(self.dim, dtype=np.int)
            for j in literals:
                self.literals[j] = 1
        else:
            # TODO: We could make some mistakes with this logic
            raise Exception("Literals must be a numpy array of ints. It must either be "
                            "binary an array of indices")

        # Make sure we have a binary conjunction
        assert self.literals.dtype == np.int
        assert np.amax(self.literals) <= 1
        assert np.amin(self.literals) >= 0

        # Set the threshold for the conjunction
        self._thresh = np.sum(self.literals)

    def evaluate(self, x):
        return np.dot(x, self.literals) >= self._thresh