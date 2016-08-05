import lasagne.layers as L
import lasagne.nonlinearities as NL
import numpy as np
import theano.tensor as T

from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.network import GRUNetwork
from rllab.core.serializable import Serializable
from rllab.distributions.recurrent_categorical import RecurrentCategorical
from rllab.misc import ext
from rllab.spaces import Discrete, Product
from rllab.misc import special
from rllab.misc.overrides import overrides
from rllab.policies.base import StochasticPolicy

from praglang.layers.decoder import GRUDecoderLayer
from praglang.spaces import DiscreteSequence


class RecurrentConversationAgentPolicy(StochasticPolicy, LasagnePowered, Serializable):
    """
    Recurrent policy which can either take an action or generate a token
    sequence at each timestep.
    """

    def __init__(
            self,
            env_spec,
            b_agent,
            hidden_sizes=(32,),
            embedding_size=32):
        """
        Args:
            env_spec: A spec for the env.
            hidden_sizes: list of sizes for the fully connected hidden layers
            hidden_nonlinearity: nonlinearity used for each hidden layer
        """
        assert isinstance(env_spec.action_space, Product)
        assert isinstance(env_spec.action_space.components[0], Discrete)
        assert isinstance(env_spec.action_space.components[1], DiscreteSequence)

        Serializable.quick_init(self, locals())
        super(RecurrentConversationAgentPolicy, self).__init__(env_spec)

        # TODO remove this requirement
        assert len(hidden_sizes) == 1

        # TODO
        input_shape = (env_spec.observation_space.flat_dim,)

        # Build input GRU network with embedding lookup.
        l_inp = L.InputLayer((None, b_agent.num_tokens, b_agent.vocab_size),
                             input_var=T.itensor3())
        l_emb = L.EmbeddingLayer(l_inp, input_size=b_agent.vocab_size,
                                 output_size=embedding_size)
        l_enc_hid = L.GRULayer(l_emb, num_units=hidden_sizes[0],
                               only_return_final=True)

        # Build output GRU network with embedding lookup.
        # This network is run every timestep when the actor chooses to make an
        # utterance.
        # TODO also take in environment state
        l_dec_out = L.DenseLayer(L.InputLayer((None, hidden_sizes[0])),
                                 num_units=b_agent.vocab_size,
                                 nonlinearity=NL.softmax, name="dec_out")
        l_dec = GRUDecoderLayer(l_enc_hid, hidden_sizes[0], b_agent.num_tokens,
                                l_emb, l_dec_out)

        self._encoder_input = l_inp
        self._decoder_output = l_dec

        self._f_prob = ext.compile_function([l_inp.input_var],
                                            L.get_output([l_dec]))

        self._prev_action = None
        self._prev_hidden = None
        self._hidden_sizes = hidden_sizes
        #self._dist = RecurrentCategorical(env_spec.action_space.n)

        self.reset()

        LasagnePowered.__init__(self, [self._decoder_output])

    @overrides
    def dist_info_sym(self, obs_var, state_info_vars):
        n_batches, n_steps = obs_var.shape[:2]
        obs_var = obs_var.reshape((n_batches, n_steps, -1))
        all_input_var = obs_var
        return dict(
            p_utterance=L.get_output(
                self._decoder_output,
                {self._encoder_input: all_input_var}
            )
        )

    def reset(self):
        # TODO
        pass

    # The return value is a pair. The first item is a matrix (N, A), where each
    # entry corresponds to the action value taken. The second item is a vector
    # of length N, where each entry is the density value for that action, under
    # the current policy
    @overrides
    def get_action(self, observation):
        all_input = self.observation_space.flatten(observation)
        out_probs = self._f_prob([all_input])
        actions = [special.weighted_sample(out_probs_t, xrange(self.b_agent.vocab_size))
                   for out_probs_t in out_probs]
        agent_info = dict(p_utterance=out_probs)
        return actions, agent_info

    @property
    @overrides
    def recurrent(self):
        return True

    @property
    def distribution(self):
        raise ValueError("called")
        return self._dist

    @property
    def state_info_keys(self):
        return []
