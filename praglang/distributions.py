from theano import tensor as T

from rllab.distributions.base import Distribution
from rllab.distributions.categorical import Categorical

TINY = 1e-8


class ConditionalChain(Distribution):
    """
    A distribution which factorizes into the form

    p_D(A) 1{A = K}(p_C(S | A = K))

    where `p_D` is a categorical distribution and `p_C` is some arbitrary
    distribution. Here the event space is a product `A X S`, where `A`
    corresponds to categorical assignments and `S` corresponds to some event
    in the chained distribution.
    """
    # TODO: Is it easier or harder to represent the distribution
    #     p_D(A_D) p_C(A_C | A_D)
    # in our use case? Find out and pick the easier one. :)

    def __init__(self, discrete_dim, chain_trigger, chain_distr):
        """
        Args:
            discrete_dim: Cardinality of the categorical distribution.
            chain_trigger: Value of the categorical distribution which should
                trigger the chained distribution.
            chain_distribution: A child `Distribution` instance which is
                triggered when the parent categorical distribution selects the
                particular value `chain_trigger`. This should be a discrete
                distribution; bad things will happen if it is not one.
        """
        self._prior_distr = Categorical(discrete_dim)
        self._chain_trigger = chain_trigger
        self._chain_distr = chain_distr

        # This is easier to code if the chain trigger is the final choice in
        # the categorical space.
        assert self._chain_trigger == discrete_dim - 1

    def kl_sym(self, p_dist_info_vars, q_dist_info_vars):
        """
        Compute symbolic KL divergence KL[P||Q], where both `P` and `Q` are
        `ConditionalChain` distributions.

        \sum_x P(x) [\log \frac{P(x)}{Q(x)}]
        """

        # Retrieve p_D(A), p_C(S) for each distribution.
        p_D_p, p_C_p = p_dist_info_vars["p_D"], p_dist_info_vars["p_C"]
        p_D_q, p_C_q = q_dist_info_vars["p_D"], q_dist_info_vars["p_C"]

        # First compute KL on assignments where A != K, i.e. the chain
        # distribution doesn't come into play
        #
        # \sum_{A: A != K} p_D(A) log(p_D(A) / q_D(A))
        kl = (p_D_p[:, :-1] * (T.log(p_D_p[:, :-1] + TINY)
                               - T.log(p_D_q[:, :-1]))).sum()

        # Now include KL on assignments where A == K
        # Retrieve P_D(A = K), Q_D(A = K), of shape (batch_size,)
        # These will be broadcast over the p_C tensors
        p_chain_p, p_chain_q = p_D_p[:, -1], p_D_q[:, -1]
        kl += (p_chain_p * p_C_p * (T.log(p_chain_p * p_C_p + TINY)
                                    - T.log(p_chain_q * p_C_q + TINY))).sum()

        return kl

    def kl(self, p_dist_info_vars, q_dist_info_vars):
        """
        Compute KL divergence KL[P||Q].
        """

        # Retrieve p_D(A), p_C(S) for each distribution.
        p_D_p, p_C_p = p_dist_info_vars["p_D"], p_dist_info_vars["p_C"]
        p_D_q, p_C_q = q_dist_info_vars["p_D"], q_dist_info_vars["p_C"]

        # First compute KL on assignments where A != K, i.e. the chain
        # distribution doesn't come into play
        #
        # \sum_{A: A != K} p_D(A) log(p_D(A) / q_D(A))
        kl = (p_D_p[:, :-1] * (np.log(p_D_p[:, :-1] + TINY)
                               - np.log(p_D_q[:, :-1]))).sum()

        # Now include KL on assignments where A == K
        # Retrieve P_D(A = K), Q_D(A = K), of shape (batch_size,)
        # These will be broadcast over the p_C tensors
        p_chain_p, p_chain_q = p_D_p[:, -1], p_D_q[:, -1]
        kl += (p_chain_p * p_C_p * (np.log(p_chain_p * p_C_p + TINY)
                                    - np.log(p_chain_q * p_C_q + TINY))).sum()

        return kl


