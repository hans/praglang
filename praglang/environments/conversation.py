from rllab.envs.base import Env, Step
from rllab.envs.grid_world_env import GridWorldEnv
from rllab.spaces import Box, Discrete, Product

from praglang.spaces import DiscreteSequence


# Constants indicating whether our agent most recently took an action in the
# wrapped environment or made an utterance.
WRAPPED, UTTER = 0, 1


class SituatedConversationEnvironment(Env):

    """
    An environment which simulates conversation between two agents A
    and B in some situated environment.

    A represents the policy being learned. It can emit token sequences
    to B and process the responses from B.
    B is some fixed external resource (i.e., not a policy being learned)
    which accepts A's utterances and returns its own utterances. These may
    be a stochastic function of A's input.

    A can also take actions in the situated environment being wrapped by this
    class. A's action space thus consists of the product of the action
    space of the wrapped environment and the utterance space.
    """

    def __init__(self, env=None, num_tokens=None, vocabulary=None,
                 b_agent=None, *args, **kwargs):
        """
        Args:
            env: Environment which is being wrapped by this conversational
                environment. This environment must have a discrete action
                space. The RL agent will be able to take any of the actions
                in this environment, or select a new action `utterance`,
                managed by this class.
            num_tokens: Maximum number of tokens in each utterance
                transmitted from A to B or B to A.
            vocab_size:
            b_agent: Function mapping from A utterance (i.e., a discrete token
                sequence encoded as a one-hot matrix) to B response (also a
                discrete token sequence encoded as a one-hot matrix).
        """
        super(SituatedConversationEnvironment, self).__init__(*args, **kwargs)

        if env is None:
            env = GridWorldEnv()
        if num_tokens is None:
            num_tokens = 5
        if vocabulary is None:
            vocabulary = ["a", "b", "c", "d"]

        assert isinstance(env.action_space, Discrete)
        self._env = env

        self.num_tokens = num_tokens
        self.vocab = vocabulary
        self.vocab_size = len(vocabulary)

        self._sequence_space = DiscreteSequence(self.vocab_size, num_tokens)

        # TODO: Should also join with observation space of wrapped env
        self._obs_space = Product(env.observation_space, self._sequence_space)

        # The agent can choose to take any action in the wrapped env or to make
        # an utterance. We add a single option to the discrete action space of
        # the wrapped env in order to represent this "utterance" choice.
        action_space = Discrete(env.action_space.n + 1)
        # When the action "utterance" is chosen, the agent also needs to
        # predict a distribution over token sequences.
        #
        # TODO: Does this play nicely with the built-in training algos, e.g.
        # when the agent is taking actions that don't make use of this data?
        # Probably not.
        self._action_space = Product(action_space, self._sequence_space)

        if b_agent is None:
            b_agent = lambda _: self._sequence_space.sample()

        self._b_agent = b_agent

    @property
    def observation_space(self):
        return self._obs_space

    @property
    def action_space(self):
        return self._action_space

    def reset(self):
        # Reset tracking state.
        self._sent, self._received = [], []
        self._wrapped_actions = []
        self._last_action = WRAPPED

        # Reset wrapped env and pull an initial observation.
        self._last_wrapped_obs = self._env.reset()

        # Sample an input sequence.
        first_seq = self._sequence_space.sample()
        self._received.append(first_seq)

        return (self._last_wrapped_obs, first_seq)

    def step(self, (action, sequence)):
        """
        Args:
            action: A token sequence emitted by `A`.
        """

        if action < self._env.action_space.n:
            self._last_action = WRAPPED
            self._wrapped_actions.append(action)

            # Agent took an action in wrapped env.
            wrapped_step = self._env.step(action)
            self._last_wrapped_obs = wrapped_step.observation

            observation = (wrapped_step.observation, self._received[-1])
            return Step(observation=observation, reward=wrapped_step.reward,
                        done=wrapped_step.done)

        # If we're here, it means the agent has chosen to make an utterance.
        self._last_action = UTTER
        # 0 reward + certainly not done.
        reward = 0.0
        done = False

        # Send the utterance to B and get a response.
        self._sent.append(sequence)
        response = self._b_agent(sequence)
        self._received.append(response)

        observation = (self._last_wrapped_obs, response)
        return Step(observation=response, reward=reward, done=done)

    def render(self):
        last_received_toks = [self.vocab[idx] for idx in self._received[-1]]
        print "B: %s" % " ".join(last_received_toks)

        if len(self._sent) == 0:
            return

        if self._last_action == WRAPPED:
            print "A took action %i" % self._wrapped_actions[-1]
        else:
            last_sent_toks = [self.vocab[idx] for idx in self._sent[-1]]
            print "A: %s" % " ".join(last_sent_toks)

