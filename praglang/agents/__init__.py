"""
This module defines fixed conversational agents for various environments.
"""


def Agent(object):

    @property
    def vocabulary(self):
        """
        A list of tokens that this agent uses in its utterances and expects in
        messages directed to it.
        """
        raise NotImplementedError

    def __call__(self, state, message):
        """
        Respond to a message, which is a sequence of tokens from this agent's
        vocabulary.

        Args:
            state: Environment state.
            message: Token sequence from another agent. A sequence of integer
                indexes into `self.vocabulary`.

        Returns:
            response: Token sequence directed at the agent who sent the
                original message.
        """
        raise NotImplementedError
