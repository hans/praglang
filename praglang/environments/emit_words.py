from collections import Counter
import sys

import numpy as np
from rllab.envs.base import Env
from rllab.envs.base import Step
from rllab.spaces import Box, Discrete


with open("wordsEn.txt", "r") as words_f:
    WORDS = [line.strip() for line in words_f]


class WordEmissionEnvironment(Env):

    sparse_rewards = False

    def __init__(self, *args, **kwargs):
        super(WordEmissionEnvironment, self).__init__(*args, **kwargs)

        self.vocab = WORDS
        self._build_prefix_tree()

        self.chars = list(set("".join(self.vocab)))
        self.char2id = {c: idx for idx, c in enumerate(self.chars)}

    def _build_prefix_tree(self):
        prefix_tree = []
        for item in self.vocab:
            prefix_tree.extend(item[:prefix] for prefix in range(1, len(item) + 1))

        self._prefix_tree = frozenset(prefix_tree)

    @property
    def observation_space(self):
        return Box(low=0, high=1, shape=(len(self.chars),))

    @property
    def action_space(self):
        return Discrete(len(self.chars))

    def log_diagnostics(self, paths):
        # Collect valid and invalid word trajectories.
        seen_valid, seen_invalid = Counter(), Counter()
        for path in paths:
            char_idx_seq = path["actions"].nonzero()[1]
            sampled_word = "".join(self.chars[idx] for idx in char_idx_seq)
            if sampled_word in self.vocab:
                seen_valid[sampled_word] += 1
            else:
                seen_invalid[sampled_word] += 1

        print "% 4i valid (% 4i unique): %s" % (sum(seen_valid.values()), len(seen_valid), seen_valid.most_common(30))
        print "% 4i invalid (% 4i unique): %s" % (sum(seen_invalid.values()), len(seen_invalid), seen_invalid.most_common(30))
        sys.stdout.flush()

    def reset(self):
        # Select a random start character.
        idx = np.random.randint(0, len(self.chars))
        self._emitted = [self.chars[idx]]

        # Observation: last output character
        obs = np.zeros((len(self.chars),))
        obs[idx] = 1
        return obs

    def step(self, action):
        self._emitted.append(self.chars[action])

        emitted_str = "".join(self._emitted)
        valid_prefix = emitted_str in self._prefix_tree

        done = False
        reward = 0.0
        if valid_prefix:
            if len(self._emitted) > 2 and emitted_str in self.vocab:
                done = True
                reward = float(len(self._emitted)) if self.sparse_rewards else 2.0
            elif not self.sparse_rewards:
                reward = 1.0
        else:
            done = True
            if not self.sparse_rewards:
                reward = -1.0

        # Sample another random observation.
        # (Recurrent policy handles the state we care about..)
        obs = np.zeros((len(self.chars),))
        obs[action] = 1

        return Step(observation=obs, reward=reward, done=done)

    def render(self):
        print "".join(self._emitted)
