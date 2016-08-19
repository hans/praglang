from collections import Counter
import sys

import numpy as np
from rllab.envs.base import Env, Step
from rllab.spaces.box import Box
from rllab.spaces.discrete import Discrete

from praglang.spaces import DiscreteSequence


class AutoencoderEnvironment(Env):

    stop_on_error = False
    sparse_rewards = True

    def __init__(self, chars="abc", max_length=3, *args, **kwargs):
        super(AutoencoderEnvironment, self).__init__(*args, **kwargs)

        self.vocab = list(chars)
        self.max_length = max_length

    @property
    def observation_space(self):
        return DiscreteSequence(len(self.vocab), self.max_length)
#        return Box(low=0, high=1, shape=(self.max_length, len(self.vocab)))

    @property
    def action_space(self):
        return Discrete(len(self.vocab))

    def log_diagnostics(self, paths):
        format_str = "%% %is => %% %is" % (self.max_length, self.max_length)

        for path in paths:
            observations, actions = path["observations"], path["actions"]

            observed_str = [self.vocab[idx] for idx
                            in self.observation_space.unflatten(observations)]
            action_str = [self.vocab[idx] for idx
                          in self.observation_space.unflatten(actions)]

            print format_str % ("".join(observed_str), "".join(action_str))

        sys.stdout.flush()

    def reset(self):
        # Sample a random input.
        length = np.random.randint(1, self.max_length + 1)
        input_ids = np.random.choice(len(self.vocab), length)

        self._emitted = []
        self._input = input_ids

        # observation is always the input sequence
        return self._input

    def step(self, action):
        self._emitted.append(action)

        reward = 0.0
        done = len(self._emitted) == len(self._input)

        expected = self._input[len(self._emitted) - 1]

        if (done or not self.sparse_rewards) and action == expected:
            reward = 1.0
        if action != expected and self.stop_on_error:
            done = True

        # Constant observation -- policy only reads this once at start
        return Step(observation=self._input, reward=reward, done=done)

    def render(self):
        input_chars = [self.vocab[idx] for idx in self._input]
        output_chars = [self.vocab[idx] for idx in self._emitted]

        print "% 3s => % 3s" % ("".join(input_chars), "".join(output_chars))
