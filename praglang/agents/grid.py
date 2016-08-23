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
        "whr", # where to go?
        "wn", "ws", "we", "ww", # destroy walls in a certain direction

        # Expected send/receive messages
        "n", "s", "e", "w", # point in direction
        "dn", "ds", "de", "dw", # notify that wall has been destroyed
    ]

    # Vocabulary: all characters in possible valid words
    vocab = list(set("".join(words)))

    num_tokens = max(len(word) for word in words)

    def __call__(self, env, message):
        # TODO: Handle padding / stop token?
        message_str = "".join(self.vocab[idx] for idx in message)

        response = ""

        # Just hard mapping for now. Yep.
        if message_str == "whr":
            # TODO find goal direction
        elif message_str.startswith("w"):
            # TODO destroy a wall
            direction = message_str[1]
            response = "d%s" % direction

        return response, 0.0


class GridWorldSlaveAgent(Agent):

    """
    A "slave" which has only partial vision of the grid world and needs to
    navigate through the grid to the goal.
    """

    pass
