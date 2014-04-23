"""
Sample the distribution over binary strings X using Metropolis-Hastings.
Statistical queries amount to expectations over the distribution and can
be approximated from the samples.
"""
import numpy as np
from utils import memoize

class Proposal(object):
    """
    Metropolis-Hastings proposal
    """
    def __init__(self):
        # Set this to true if the log prob of proposing xf from x0 is the
        # same as proposing x0 from xf.
        self._is_symmetric = False

    def propose(self, x0):
        """
        Return a new state proposal
        """
        return None

    def log_p(self, x0, xf):
        """
        Log probability of proposing to move from x0 to xf
        """
        return -np.Inf

    @property
    def is_symmetric(self):
        return self._is_symmetric


class FlipOneBitProposal(Proposal):
    """
    Propose to flip one bit of the sample
    """
    def __init__(self, dim):
        super(FlipOneBitProposal, self).__init__()
        self.dim = dim
        self._is_symmetric = True

    def propose(self, x0):
        """
        Propose a new state xf with Hamming distance 1 from x0
        """
        # assert x0.dtype == np.bool
        # TODO: This could be faster if we represented the input as an int
        # and used xf = x0 ^ (1 << b)
        b = np.random.randint(0, self.dim)
        xf = x0.copy()
        xf[b] = not xf[b]
        return xf

    def log_p(self, x0, xf):
        return -np.log(self.dim)


class MetropolisHastingsSampler:
    """
    A Markov chain to sample a given distribution using MH
    """
    def __init__(self, distribution, proposal, x0, n_samples=1000):
        self.distribution = distribution
        self.proposal = proposal
        self._is_symmetric = proposal.is_symmetric

        # Allocate space for the samples
        self.samples = np.zeros((n_samples, self.distribution.dim))

        # Set the first state
        self.samples[0, :] = x0
        self.offset = 0

    def step(self, n_steps=1):
        """
        Take specified number of MH steps
        """
        for s in np.arange(n_steps):
            self._single_step()

    def _single_step(self):
        """
        Take a single step of the Markov chain
        """
        # Get current state
        x0 = self.samples[self.offset, :]

        # Propose a new state
        xf = self.proposal.propose(x0)

        # Evaluate the ratio of proposal probabilities
        if self._is_symmetric:
            prop_ratio = 0.0
        else:
            prop_ratio = self.proposal.log_p(xf, x0) - self.proposal.log_p(x0, xf)

        # Evaluate the ratio of the probabilities under the distribution
        dist_ratio = self.distribution.log_p(xf) - self.distribution.log_p(x0)

        # Take step
        if np.log(np.random.rand()) < dist_ratio + prop_ratio:
            # Accept proposal
            self._append_sample(xf)
        else:
            # Reject (stay in same state)
            self._append_sample(x0)

    def _append_sample(self, x):
        """
        Append a new state to the sample
        """
        self.offset += 1
        try:
            self.samples[self.offset, :] = x
        except:
            # We overflowed the buffer. Double it.
            self.samples = np.concatenate((self.samples, np.zeros_like(self.samples)))
            self.samples[self.offset, :] = x

    def mean(self, f=None):
        """
        Compute the mean of f(x) for the samples x_1, ...
        """
        if f is None:
            # By default, return the mean of the samples
            return np.mean(self.samples[:self.offset,:], axis=0)
        else:
            f_samples = np.apply_along_axis(f, 1, self.samples[:self.offset, :])
            return np.mean(f_samples, axis=0)