"""
Test cases
"""
import numpy as np
from scipy.misc import logsumexp

from approx_distribution import ExponentialWeightsDistribution
from mh_sampler import FlipOneBitProposal, MetropolisHastingsSampler
from queries import Conjunction

def initialize_randomness(seed):
    """
    Initialize the random num generator for reproducibility
    """

def test_sampler_e2e():
    """
    End-to-end test for the sampler with an exponential weights distribution
    Define a distribution over 4 bits so we can easily calculate the true probability
    of each of the 16 states. Use MH to sample the space of inputs and evaluate the
    expectation of a query under the current distribution.
    """
    dim = 4
    n_samples = 10000
    queries = [Conjunction(dim, np.array([0,0,1,0])),
               Conjunction(dim, np.array([0,1,0,0]))]
    weights = np.ones(len(queries))
    distribution = ExponentialWeightsDistribution(dim, queries, weights=weights)
    proposal = FlipOneBitProposal(dim)
    sampler = MetropolisHastingsSampler(distribution, proposal, np.zeros(dim), n_samples=n_samples)

    # Run the sampler
    print "Running sampler"
    sampler.step(n_steps=n_samples)

    # Compute the empirical distribution
    print "Computing empirical and true distribution"
    p_empirical = np.zeros(2**dim)
    log_p_true = np.zeros(2**dim)
    for d in np.arange(2**dim):
        # Get a bool array representing d in binary
        sb = np.binary_repr(d, width=dim)
        b = np.array([np.int(sbi) for sbi in sb])

        # Define an identity function for x==b
        id_b = lambda x: np.allclose(x,b)

        # Empirical probability of the string is simply the number of times we sample it
        p_empirical[d] = sampler.mean(id_b)

        # Compute the true log probability
        log_p_true[d] = distribution.log_p(b)

    p_true = np.exp(log_p_true - logsumexp(log_p_true))

    print "p_true: %s" % str(p_true)
    print "p_empirical: %s" % str(p_empirical)
    print "||p_empirical - p_true||_1 = %.3f" % np.sum(np.abs(p_empirical-p_true))

if __name__ == "__main__":
    test_sampler_e2e()
