"""
A conversational agent which operates in the `grid` environment.
"""

from praglang.agents import Agent


class GridWorldMasterAgent(Agent):

    """
    A "master" which can see the entire grid world and perform certain
    executive-level actions on the world (e.g. modify the map), but does not
    actually exist in the world.
    """

    words = [
        # Expected receive messages
        "wh", # where to go?
        "wn", "ws", "we", "ww", # destroy walls in a certain direction

        # Expected send/receive messages
        "n", "s", "e", "w", # point in direction
        "wn", "ws", "we", "ww", # notify that wall has been destroyed
    ]

    # Vocabulary: all characters in possible valid words
    vocab = list(set("".join(words)))

    token2idx = {token: idx for idx, token in enumerate(vocab)}

    num_tokens = max(len(word) for word in words)

    directions = {
        "n": [0, 1],
        "s": [0, -1],
        "e": [1, 0],
        "w": [-1, 0],
    }
    directions = {k: np.array(v) for k, v in directions.items()}

    def __call__(self, env, message):
        # TODO: Handle padding / stop token?
        message_str = "".join(self.vocab[idx] for idx in message)

        response = ""
        reward = 0.0

        slave_x = env.state / env.n_col
        slave_y = env.state % env.n_col
        slave_coords = np.array([slave_x, slave_y])

        # Just hard mapping for now. Yep.
        if message_str == "whr":
            goal_x = env.goal_state / env.n_col
            goal_y = env.goal_state % env.n_col
            goal_coords = np.array([goal_x, goal_y])

            valid_directions = []
            for direction, increment in self.directions.items():
                cell_coords = slave_coords + increment
                cell_type = env.map_desc[cell_coords]
                if cell_type not in ["W", "H"]:
                    # Score by Manhattan distance
                    distance = np.abs(goal_coords - slave_coords).sum()
                    valid_directions.append((direction, distance))

            if not valid_directions:
                # Stuck!
                raise RuntimeError

            best_direction = min(valid_directions, key=lambda d, v: v)[0]
            response = best_direction
        elif message_str.startswith("w"):
            direction = message_str[1]
            increment = self.directions[direction]

            # Calculate indicated point on map and retrieve the cell type
            point_coords = np.clip(slave_coords + increment,
                                   [0, 0], env.n_row - 1, env.n_col - 1])
            point_type = env.map_desc[point_coords[0], point_coords[1]]

            if point_type == "W":
                env.map_desc[point_coords] = "F"
                response = message_str
            else:
                # Slave asked for a wall destruction when there was no wall in
                # the specified direction.
                reward = -1.0

        response = [token2idx[token] for token in response]
        return response, reward


class GridWorldSlaveAgent(Agent):

    """
    A "slave" which has only partial vision of the grid world and needs to
    navigate through the grid to the goal.
    """

    pass
