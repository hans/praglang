"""
A conversational agent which operates in the `grid` environment.
"""

from praglang.agents import Agent


class GridWorldAgent(Agent):

    words = [
        # Expected receive messages
        "whr",
        # Expected send/receive messages
        "n", "s", "e", "w"
    ]

    # Vocabulary: all characters in possible valid words
    vocab = list(set("".join(words)))

    num_tokens = max(len(word) for word in words)

    def __call__(self, env, message):
        # TODO: Handle padding / stop token?
        message_str = "".join(self.vocab[idx] for idx in message)

        # TODO: build response

        return message, 0.0
