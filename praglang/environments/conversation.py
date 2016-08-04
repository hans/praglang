from rllab.envs.base import Env, Step
from rllab.spaces import Box, Discrete, Product

from praglang.spaces import DiscreteSequence


class ConversationEnvironment(Env):

    """
    An environment which simulates conversation between two agents A
    and B.

    A represents the policy being learned. It can emit token sequences
    to B and process the responses from B.
    B is some fixed external resource (i.e., not a policy being learned)
    which accepts A's utterances and returns its own utterances. These may
    be a stochastic function of A's input.

    We expect that agent A has something like a parametric encoder-decoder
    policy.
    """

    def __init__(self, num_tokens=None, vocabulary=None, b_agent=None, *args, **kwargs):
        """
        Args:
            num_tokens: Maximum number of tokens in each utterance
                transmitted from A to B or B to A.
            vocab_size:
            b_agent: Function mapping from A utterance (i.e., a discrete token
                sequence encoded as a one-hot matrix) to B response (also a
                discrete token sequence encoded as a one-hot matrix).
        """
        super(ConversationEnvironment, self).__init__(*args, **kwargs)

        if num_tokens is None:
            num_tokens = 5
        if vocabulary is None:
            vocabulary = ["a", "b", "c", "d"]

        self.num_tokens = num_tokens
        self.vocab = vocabulary
        self.vocab_size = len(vocabulary)

        self._obs_space = DiscreteSequence(self.vocab_size, num_tokens)
        self._action_space = DiscreteSequence(self.vocab_size, num_tokens)

        if b_agent is None:
            b_agent = lambda _: self._action_space.sample()

        self._b_agent = b_agent

    @property
    def observation_space(self):
        return self._obs_space

    @property
    def action_space(self):
        return self._action_space

    def reset(self):
        self._sent, self._received = [], []
        return self._action_space.sample()

    def step(self, action):
        """
        Args:
            action: A token sequence emitted by `A`.
        """

        # TODO reward function / objective
        # Should probably come via dependency injection for simplicity
        reward = 0.0
        done = False

        # Send the utterance to B and get a response.
        self._sent.append(action)
        response = self._b_agent(action)
        self._received.append(response)

        return Step(observation=response, reward=reward, done=done)

    def render(self):
        if len(self._sent) == 0:
            return

        last_sent_toks = [self.vocab[idx] for idx in self._sent[-1]]
        print "A: %s" % " ".join(last_sent_toks)

        last_received_toks = [self.vocab[idx] for idx in self._received[-1]]
        print "B: %s" % " ".join(last_received_toks)
