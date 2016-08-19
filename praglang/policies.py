import lasagne.layers as L
import lasagne.nonlinearities as NL
import numpy as np
import theano
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
from praglang.util import GRUStepLayer


class RecurrentCategoricalPolicy(StochasticPolicy, LasagnePowered, Serializable):
    """
    Customized form of `rllab.policies.categorical_gru_policy.CategoricalGRUPolicy`.
    Supports temperature sampling.
    """

    def __init__(
            self,
            env_spec,
            hidden_sizes=(32,),
            state_include_action=True,
            hidden_nonlinearity=NL.tanh,
            temperature=1.0):
        """
        :param env_spec: A spec for the env.
        :param hidden_sizes: list of sizes for the fully connected hidden layers
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :return:
        """
        assert isinstance(env_spec.action_space, Discrete)
        Serializable.quick_init(self, locals())
        super(RecurrentCategoricalPolicy, self).__init__(env_spec)

        assert len(hidden_sizes) == 1

        if state_include_action:
            input_shape = (env_spec.observation_space.flat_dim + env_spec.action_space.flat_dim,)
        else:
            input_shape = (env_spec.observation_space.flat_dim,)

        temperature_softmax = lambda logits: NL.softmax(1.0 / temperature * logits)

        prob_network = GRUNetwork(
            input_shape=input_shape,
            output_dim=env_spec.action_space.n,
            hidden_dim=hidden_sizes[0],
            hidden_nonlinearity=hidden_nonlinearity,
            output_nonlinearity=temperature_softmax,
        )

        self._prob_network = prob_network
        self._state_include_action = state_include_action

        self._f_step_prob = ext.compile_function(
            [
                prob_network.step_input_layer.input_var,
                prob_network.step_prev_hidden_layer.input_var
            ],
            L.get_output([
                prob_network.step_output_layer,
                prob_network.step_hidden_layer
            ])
        )

        self._prev_action = None
        self._prev_hidden = None
        self._hidden_sizes = hidden_sizes
        self._dist = RecurrentCategorical(env_spec.action_space.n)

        self.reset()

        LasagnePowered.__init__(self, [prob_network.output_layer])

    @overrides
    def dist_info_sym(self, obs_var, state_info_vars):
        n_batches, n_steps = obs_var.shape[:2]
        obs_var = obs_var.reshape((n_batches, n_steps, -1))
        if self._state_include_action:
            prev_action_var = state_info_vars["prev_action"]
            all_input_var = TT.concatenate(
                [obs_var, prev_action_var],
                axis=2
            )
        else:
            all_input_var = obs_var
        return dict(
            prob=L.get_output(
                self._prob_network.output_layer,
                {self._prob_network.input_layer: all_input_var}
            )
        )

    def reset(self):
        self._prev_action = None
        self._prev_hidden = self._prob_network.hid_init_param.get_value()

    # The return value is a pair. The first item is a matrix (N, A), where each
    # entry corresponds to the action value taken. The second item is a vector
    # of length N, where each entry is the density value for that action, under
    # the current policy
    @overrides
    def get_action(self, observation):
        if self._state_include_action:
            if self._prev_action is None:
                prev_action = np.zeros((self.action_space.flat_dim,))
            else:
                prev_action = self.action_space.flatten(self._prev_action)
            all_input = np.concatenate([
                self.observation_space.flatten(observation),
                prev_action
            ])
        else:
            all_input = self.observation_space.flatten(observation)
            # should not be used
            prev_action = np.nan
        probs, hidden_vec = [x[0] for x in self._f_step_prob([all_input], [self._prev_hidden])]
        action = special.weighted_sample(probs, xrange(self.action_space.n))
        self._prev_action = action
        self._prev_hidden = hidden_vec
        agent_info = dict(prob=probs)
        if self._state_include_action:
            agent_info["prev_action"] = prev_action
        return action, agent_info

    @property
    @overrides
    def recurrent(self):
        return True

    @property
    def distribution(self):
        return self._dist

    @property
    def state_info_keys(self):
        if self._state_include_action:
            return ["prev_action"]
        else:
            return []


class EncoderDecoderPolicy(StochasticPolicy, LasagnePowered, Serializable):

    def __init__(
            self,
            env_spec,
            num_timesteps,
            vocab_size,
            hidden_sizes=(64,),
            embedding_size=32,
            hidden_nonlinearity=NL.tanh):
        """
        :param env_spec: A spec for the env.
        :param hidden_sizes: list of sizes for the fully connected hidden layers
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :return:
        """
        assert isinstance(env_spec.observation_space, DiscreteSequence)
        assert isinstance(env_spec.action_space, Discrete)
        Serializable.quick_init(self, locals())
        super(EncoderDecoderPolicy, self).__init__(env_spec)

        assert len(hidden_sizes) == 1
        self.hidden_sizes = hidden_sizes

        # Build input GRU network with embedding lookup.
        l_inp = L.InputLayer((None, num_timesteps),
                             input_var=T.imatrix(), name="enc_inp")
        l_emb = L.EmbeddingLayer(l_inp, input_size=vocab_size,
                                 output_size=embedding_size, name="enc_emb")
        l_enc_hid = L.GRULayer(l_emb, num_units=hidden_sizes[0],
                               only_return_final=True, name="enc_gru")

        self._encoder_input = l_inp
        self._encoder_output = l_enc_hid

        # Build GRU decoder.
        l_dec_hid_init = L.InputLayer((None, hidden_sizes[0]),
                                      input_var=T.matrix())
        l_dec_out = L.DenseLayer(L.InputLayer((None, hidden_sizes[0])),
                                 num_units=vocab_size,
                                 nonlinearity=NL.softmax,
                                 name="dec_out")
        l_decoder = GRUDecoderLayer(l_dec_hid_init, hidden_sizes[0],
                                    num_timesteps, l_emb, l_dec_out)

        l_decode_step = l_decoder.get_step_layer()

        self._l_dec_hid_init = l_dec_hid_init
        self._l_decoder = l_decoder

        self._f_encode = ext.compile_function(
                [l_inp.input_var],
                L.get_output(l_enc_hid))
        self._f_decode_step = ext.compile_function(
                [l_decode_step.x_in.input_var,
                 l_decode_step.h_prev_in.input_var],
                L.get_output(l_decode_step))

        self._prev_action = None
        self._encoded = None
        self._prev_hidden = None
        self._hidden_sizes = hidden_sizes
        self._dist = RecurrentCategorical(env_spec.action_space.n)

        self.reset()

        LasagnePowered.__init__(self, [l_enc_hid, l_dec_out])

    @overrides
    def dist_info_sym(self, obs_var, state_info_vars):
        print "===== DIST INFO SYM called"
        for k, v in state_info_vars.iteritems():
            print "\t", k, v, v.ndim
        print "====="


        prev_action = state_info_vars["prev_action"]
        prev_hidden = state_info_vars["prev_hidden"]

        # Encode input sequence
        # HACK: It's the same at all timesteps; just grab first timestep value
        obs_var = obs_var[:, 0, :]
        enc_hid_mem = L.get_output(
                self._encoder_output,
                {self._encoder_input: obs_var})

        probs = L.get_output(self._l_decoder,
                            { self._l_dec_hid_init: enc_hid_mem })

        return { "prob": probs }

    def reset(self):
        self._prev_action = None
        self._encoded = None
        self._prev_hidden = None

    @overrides
    def get_action(self, observation):
        if self._prev_action is None:
            self._prev_action = 0
        if self._encoded is None:
            # Encode input sequence into batch of vectors
            self._encoded = self._f_encode([observation])[0]
        if self._prev_hidden is None:
            self._prev_hidden = self._encoded

        ret = self._f_decode_step([self._prev_action],
                                  [self._prev_hidden])
        probs, hidden_vec = [x[0] for x in ret]

        action = special.weighted_sample(probs, xrange(self.action_space.n))
        self._prev_action = action
        self._prev_hidden = hidden_vec

        agent_info = {
            "prob": probs,
            "prev_action": action,
            "prev_hidden": hidden_vec,
        }

        return action, agent_info

    @property
    @overrides
    def recurrent(self):
        return True

    @property
    def distribution(self):
        return self._dist

    @property
    def state_info_keys(self):
        return ["prev_action", "prev_hidden"]

    @property
    def state_info_metadata(self):
        return [
            (1, "int32"),
            (2, theano.config.floatX)
        ]


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
