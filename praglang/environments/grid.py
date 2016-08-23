import random

import numpy as np

from rllab.envs.base import Env, Step
from rllab.core.serializable import Serializable
from sandbox.rocky.tf.spaces.box import Box
from sandbox.rocky.tf.spaces.discrete import Discrete

MAPS = {
    "chain": [
        "GFFFFFFFFFFFFFSFFFFFFFFFFFFFG"
    ],
    "walled_chain": [
        ["GWSWF"],
        ["FWSWG"],
    ],
    "3x3": [
        [
            "FFF",
            "FSF",
            "FFG",
        ],
        [
            "FFF",
            "FSF",
            "FGF",
        ],
        [
            "FFF",
            "FSF",
            "GFF",
        ],
        [
            "FFF",
            "FSG",
            "FFF",
        ],
        [
            "FFF",
            "GSF",
            "FFF",
        ],
        [
            "FFG",
            "FSF",
            "FFF",
        ],
        [
            "FGF",
            "FSF",
            "FFF",
        ],
        [
            "GFF",
            "FSF",
            "FFF",
        ],
    ],
    "4x4_safe": [
        "SFFF",
        "FWFW",
        "FFFW",
        "WFFG"
    ],
    "4x4": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG"
    ],
}


class GridWorldEnv(Env, Serializable):
    """
    Modified version of `rllab.envs.grid_world_env.GridWorldEnv` which supports
    libraries of different graphs (rather than a single graph per
    architecture).

    S : starting point
    F : free space
    W : wall
    H : hole (terminates episode)
    G : goal

    """

    def __init__(self, desc_str='4x4'):
        Serializable.quick_init(self, locals())
        self.desc_str = desc_str
        # Map will be loaded in `self.reset`

        sample_map = MAPS[self.desc_str]
        if isinstance(sample_map[0], list):
            sample_map = sample_map[0]
        self.n_row, self.n_col = np.array(map(list, sample_map)).shape

        self.state = None
        self.domain_fig = None

    def reset(self):
        fetched_map = MAPS[self.desc_str]
        if isinstance(fetched_map[0], list):
            # We've fetched a list of maps. Sample one.
            fetched_map = random.choice(fetched_map)

        self.map_desc = np.array(map(list, fetched_map))
        assert self.map_desc.shape == (self.n_row, self.n_col)
        (start_x,), (start_y,) = np.nonzero(self.map_desc == "S")
        self.start_state = start_x * self.n_col + start_y

        (goal_x,), (goal_y,) = np.nonzero(self.map_desc == "G")
        self.goal_state = goal_x * self.n_col + goal_y

        self.state = self.start_state
        return self.get_observation()

    @staticmethod
    def action_from_direction(d):
        """
        Return the action corresponding to the given direction. This is a helper method for debugging and testing
        purposes.
        :return: the action index corresponding to the given direction
        """
        return dict(
            left=0,
            down=1,
            right=2,
            up=3
        )[d]

    def step(self, action):
        """
        action map:
        0: left
        1: down
        2: right
        3: up
        :param action: should be a one-hot vector encoding the action
        :return:
        """
        possible_next_states = self.get_possible_next_states(self.state, action)

        probs = [x[1] for x in possible_next_states]
        next_state_idx = np.random.choice(len(probs), p=probs)
        next_state = possible_next_states[next_state_idx][0]

        next_x = next_state / self.n_col
        next_y = next_state % self.n_col

        next_state_type = self.map_desc[next_x, next_y]
        if next_state_type == 'H':
            done = True
            reward = 0
        elif next_state_type in ['F', 'S']:
            done = False
            reward = 0
        elif next_state_type == 'G':
            done = True
            reward = 100
        else:
            raise NotImplementedError
        self.state = next_state

        return Step(observation=self.get_observation(), reward=reward, done=done)

    def get_possible_next_states(self, state, action):
        """
        Given the state and action, return a list of possible next states and their probabilities. Only next states
        with nonzero probabilities will be returned
        :param state: start state
        :param action: action
        :return: a list of pairs (s', p(s'|s,a))
        """

        x = state / self.n_col
        y = state % self.n_col
        coords = np.array([x, y])

        increments = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])
        next_coords = np.clip(
            coords + increments[action],
            [0, 0],
            [self.n_row - 1, self.n_col - 1]
        )
        next_state = next_coords[0] * self.n_col + next_coords[1]
        state_type = self.map_desc[x, y]
        next_state_type = self.map_desc[next_coords[0], next_coords[1]]
        if next_state_type == 'W' or state_type == 'H' or state_type == 'G':
            return [(state, 1.)]
        else:
            return [(next_state, 1.)]

    @property
    def action_space(self):
        return Discrete(4)

    @property
    def observation_space(self):
        return Discrete(self.n_row * self.n_col)

    def get_observation(self):
        return self.state


class SlaveGridWorldEnv(GridWorldEnv):

    """
    A grid-world setup where the slave is the agent in the grid world.

    Observations are limited to the immediate surroundings of the slave.
    """

    # Cell types
    CELL_TYPE_IDS = {
        "G": 0, # goal
        "W": 1, # wall
        "H": 2, # hole
        "F": 3, # free
    }
    N_TYPES = len(CELL_TYPE_IDS)

    # Directions from which to read neighbor cell data
    INCREMENTS = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])

    @property
    def observation_space(self):
        return Box(low=0., high=1.,
                   shape=(len(self.INCREMENTS), self.N_TYPES))

    def get_observation(self):
        # Get state of cells in immediate surrounding.
        x, y = self.state / self.n_col, self.state % self.n_col
        coords = np.array([x, y])

        observation = np.zeros((len(self.INCREMENTS), self.N_TYPES))
        for i, increment in enumerate(self.INCREMENTS):
            neighbor_coords = coords + increment
            try:
                cell_type = self.map_desc[neighbor_coords[0], neighbor_coords[1]]
                cell_type = self.CELL_TYPE_IDS[cell_type]
            except IndexError:
                # Out of bounds.
                continue
            except KeyError:
                # Not a cell type we are tracking.
                continue

            observation[i, cell_type] = 1

        return observation
