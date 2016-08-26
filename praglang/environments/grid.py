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
    "lopsided_chain": [
        # very easy to solve if you just ask your mother which direction the
        # goal is!
        ["GFFFFSFFFFF"],
        ["FFFFFSFFFFG"],
    ],
    "walled_chain": [
        [
            "WWWWW",
            "WWWWW",
            "GWSWF",
            "WWWWW",
            "WWWWW",
        ],
        [
            "WWWWW",
            "WWWWW",
            "FWSWG",
            "WWWWW",
            "WWWWW",
        ],
        [
            "WWGWW",
            "WWWWW",
            "WWSWW",
            "WWWWW",
            "WWFWW",
        ],
        [
            "WWFWW",
            "WWWWW",
            "WWSWW",
            "WWWWW",
            "WWGWW",
        ],
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

    def __init__(self, desc_str='4x4', goal_reward=10.0):
        Serializable.quick_init(self, locals())
        self.desc_str = desc_str
        # Map will be loaded in `self.reset`

        self.n_col, self.n_row = np.array(map(list, self._fetch_map())).shape

        self.state = None
        self.goal_reward = goal_reward

    def _fetch_map(self):
        if isinstance(self.desc_str, list):
            return self.desc_str

        sample_map = MAPS[self.desc_str]
        if isinstance(sample_map[0], list):
            # We've fetched a list of maps. Sample one.
            sample_map = random.choice(sample_map)
        return sample_map

    def reset(self):
        fetched_map = self._fetch_map()

        self.map_desc = np.array(map(list, fetched_map)).T
        assert self.map_desc.shape == (self.n_row, self.n_col)
        (start_x,), (start_y,) = np.nonzero(self.map_desc == "S")
        self.start_state = np.array([start_x, start_y])

        (goal_x,), (goal_y,) = np.nonzero(self.map_desc == "G")
        self.goal_state = np.array([goal_x, goal_y])

        self.state = self.start_state
        return self.get_observation()

    actions = [[-1,  0], # west
               [ 1,  0], # east
               [ 0, -1], # south
               [ 0,  1]] # north
    actions = np.array(actions)
    action_names = ["west", "east", "south", "north"]

    def step(self, action):
        """
        :param action: should be a one-hot vector encoding the action
        :return:
        """
        next_state = self.get_next_state(self.state, action)

        next_x, next_y = next_state
        next_state_type = self.map_desc[next_x, next_y]

        if next_state_type == 'H':
            done = True
            reward = 0
        elif next_state_type in ['F', 'S']:
            done = False
            reward = 0
        elif next_state_type == 'G':
            done = True
            reward = self.goal_reward
        else:
            raise NotImplementedError
        self.state = next_state

        return Step(observation=self.get_observation(), reward=reward, done=done)

    def bounded_increment(self, state, increment):
        return np.clip(state + increment,
                       [0, 0],
                       [self.n_row - 1, self.n_col - 1])

    def get_next_state(self, state, action):
        """
        Given the state and action, return the next state.
        :param state: start state
        :param action: action (int or ndarray increment)
        :return: a list of pairs (s', p(s'|s,a))
        """

        if isinstance(action, int):
            increment = self.actions[action]
        else:
            increment = action

        x, y = state
        next_state = self.bounded_increment(state, increment)

        state_type = self.map_desc[x, y]
        next_state_type = self.map_desc[next_state[0], next_state[1]]

        if next_state_type == 'W' or state_type == 'H' or state_type == 'G':
            return state
        else:
            return next_state

    @property
    def action_space(self):
        return Discrete(len(self.actions))

    @property
    def observation_space(self):
        return Discrete(self.n_row * self.n_col)

    def get_observation(self):
        x, y = self.state
        return self.y * self.n_col + self.x


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

    @property
    def observation_space(self):
        return Box(low=0., high=1.,
                   shape=(len(self.actions), self.N_TYPES))

    def get_observation(self):
        # Get state of cells in immediate surrounding.
        x, y = self.state

        observation = np.zeros((len(self.actions), self.N_TYPES))
        for i, increment in enumerate(self.actions):
            neighbor_coords = np.clip(
                    self.state + increment,
                    [0, 0],
                    [self.n_row - 1, self.n_col - 1]
            )

            try:
                cell_type = self.map_desc[neighbor_coords[0], neighbor_coords[1]]
                cell_type = self.CELL_TYPE_IDS[cell_type]
            except KeyError:
                # Not a cell type we are tracking.
                continue

            observation[i, cell_type] = 1

        return observation
