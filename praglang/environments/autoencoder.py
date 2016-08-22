from collections import Counter
import sys

import numpy as np
from rllab.envs.base import Env, Step
from sandbox.rocky.tf.spaces.discrete import Discrete

from praglang.spaces import DiscreteSequence



class BagAutoencoderEnvironment(Env):

    stop_on_error = False

    # Only supports sparse rewards
    sparse_rewards = True

    def __init__(self, chars="abc", max_length=3, *args, **kwargs):
        super(BagAutoencoderEnvironment, self).__init__(*args, **kwargs)

        self.vocab = list(chars)
        self.max_length = max_length

        # Need to be able to form a sequence of max length without repeats
        assert len(self.vocab) >= self.max_length

    @property
    def observation_space(self):
        return DiscreteBinaryBag(len(self.vocab))

    @property
    def action_space(self):
        return Discrete(len(self.vocab))

    def reset(self):
        # Sample length, then randomly pick chars.
        length = np.random.randint(1, self.max_length + 1)

        input_val = np.zeros((len(self.vocab),))
        idxs = np.random.choice(len(self.vocab), replace=False, size=length)
        input_val[idxs] = 1

        self._input = input_val
        self._reference = set([self.vocab[idx] for idx in idxs])
        self._emitted = []
        return self._input

    def step(self, action):
        self._emitted.append(action)

        done = len(self._emitted) == len(self._input)
        if self.stop_on_error:
            done = done or action not in self._reference

        reward = 0.0
        if done:
            reward = float(set(self._emitted) == self._reference)

        return Step(observation=self._input, reward=reward, done=done)

    def render(self):
        output_chars = [self.vocab[idx] for idx in self._emitted]

        print "% 3s\t=>\t% 3s" % ("".join(sorted(self._reference)),
                                  "".join(sorted(output_chars)))


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

    @property
    def action_space(self):
        return Discrete(len(self.vocab))

    def log_diagnostics(self, paths):
        format_str = "%% %is => %% %is" % (self.max_length, self.max_length)

        for path in paths:
            observations, actions = path["observations"], path["actions"]

            observed_str = [self.vocab[idx] for idx in observations[0]]
            action_str = [self.vocab[self.action_space.unflatten(action)]
                          for action in actions]

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
