"""
A conversational agent which operates in the `grid` environment.
"""

import numpy as np

from praglang.agents import Agent
from praglang.environments.grid import GridWorldEnv


class GridWorldMasterAgent(Agent):

    """
    A "master" which can see the entire grid world and perform certain
    executive-level actions on the world (e.g. modify the map), but does not
    actually exist in the world.
    """

    words = [
        # Expected receive messages
        "h", # where to go?
        "n", "s",
        "e", "w", # destroy walls in a certain direction

        # Expected send/receive messages
        "n", "s",
        "e", "w", # point in direction
    ]

    # Vocabulary: all characters in possible valid words
    vocab = list(set("".join(words)))

    token2idx = {token: idx for idx, token in enumerate(vocab)}

    num_tokens = max(len(word) for word in words)

    def __init__(self, env, match_reward=1.0):
        self._env = env
        self.match_reward = match_reward
        assert len(env.actions) == 4
        self.directions = dict(zip(["w", "e", "s", "n"], env.actions))

    def reset(self):
        self.received_where_query = False

    def __call__(self, env, message):
        assert isinstance(env, GridWorldEnv)
        assert env == self._env

        message_str = "".join(self.vocab[idx] for idx in message)

        response = ""
        matched = False # Did the message make any sense?
        reward = 0.0

        slave_coords = env.state

        # Just hard mapping for now. Yep.
        if message_str.startswith("h"):
            matched = not self.received_where_query
            self.received_where_query = True

            valid_directions = []
            for direction, increment in self.directions.items():
                cell_coords = env.bounded_increment(slave_coords, increment)
                try:
                    cell_type = env.map_desc[cell_coords[0], cell_coords[1]]
                except IndexError:
                    # Out of bounds.
                    continue

                if cell_type not in ["H"]:
                    # Score by Manhattan distance
                    distance = np.abs(env.goal_state - cell_coords).sum()
                    valid_directions.append((direction, distance))

            if not valid_directions:
                # Stuck!
                raise RuntimeError

            best_direction = min(valid_directions, key=lambda (d, v): v)[0]
            response = best_direction
        elif message_str.startswith(tuple(self.directions.keys())):
            matched = True

            direction = message_str[0]
            increment = self.directions.get(direction, None)
            if increment is not None:
                # Calculate indicated point on map and retrieve the cell type
                point_coords = env.bounded_increment(slave_coords, increment)
                point_type = env.map_desc[point_coords[0], point_coords[1]]

                if point_type == "W":
                    # Only match if destroying the wall actually helps us reach
                    # our goal in the map.
                    #
                    # TODO: Assumes a simple map in which destroying a wall can
                    # only be beneficial if, after moving into that spot, we've
                    # made an improvement w.r.t. distance from the goal.
                    delta = env.get_improvement(point_coords) \
                            - env.get_improvement(slave_coords)
                    if delta >= 0:
                        env.map_desc[point_coords[0], point_coords[1]] = "F"
                        response = direction
                    else:
                        matched = False
                else:
                    # Slave asked for a wall destruction when there was no wall
                    # in the specified direction.
                    matched = False

        if matched:
            reward += self.match_reward

        response = [self.token2idx[token] for token in response]
        return response, reward


class GridWorldSlaveAgent(Agent):

    """
    A "slave" which has only partial vision of the grid world and needs to
    navigate through the grid to the goal.
    """

    pass
