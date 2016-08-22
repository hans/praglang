import numpy as np
import tensorflow as tf

from rllab.core.serializable import Serializable
from rllab.misc import special
from rllab.misc.overrides import overrides
from sandbox.rocky.tf.core import layers as L
from sandbox.rocky.tf.core.layers_powered import LayersPowered
from sandbox.rocky.tf.core.network import GRUNetwork
from sandbox.rocky.tf.distributions.recurrent_categorical import RecurrentCategorical
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.policies.base import StochasticPolicy
from sandbox.rocky.tf.spaces.discrete import Discrete
from sandbox.rocky.tf.spaces.product import Product

from praglang.layers.decoder import GRUDecoderLayer
from praglang.spaces import DiscreteSequence
from praglang.util import GRUStepLayer


class RecurrentCategoricalPolicy(StochasticPolicy, LayersPowered, Serializable):

    def __init__(
            self,
            name,
            env_spec,
            hidden_dim=32,
            feature_network=None,
            state_include_action=True,
            hidden_nonlinearity=tf.tanh):
        """
        :param env_spec: A spec for the env.
        :param hidden_dim: dimension of hidden layer
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :return:
        """
        with tf.variable_scope(name):
            assert isinstance(env_spec.action_space, Discrete)
            Serializable.quick_init(self, locals())
            super(RecurrentCategoricalPolicy, self).__init__(env_spec)

            obs_dim = env_spec.observation_space.flat_dim
            action_dim = env_spec.action_space.flat_dim

            if state_include_action:
                input_dim = obs_dim + action_dim
            else:
                input_dim = obs_dim

            l_input = L.InputLayer(
                shape=(None, None, input_dim),
                name="input"
            )

            if feature_network is None:
                feature_dim = input_dim
                l_flat_feature = None
                l_feature = l_input
            else:
                feature_dim = feature_network.output_layer.output_shape[-1]
                l_flat_feature = feature_network.output_layer
                l_feature = L.OpLayer(
                    l_flat_feature,
                    extras=[l_input],
                    name="reshape_feature",
                    op=lambda flat_feature, input: tf.reshape(
                        flat_feature,
                        tf.pack([tf.shape(input)[0], tf.shape(input)[1], feature_dim])
                    ),
                    shape_op=lambda _, input_shape: (input_shape[0], input_shape[1], feature_dim)
                )

            prob_network = GRUNetwork(
                input_shape=(feature_dim,),
                input_layer=l_feature,
                output_dim=env_spec.action_space.n,
                hidden_dim=hidden_dim,
                hidden_nonlinearity=hidden_nonlinearity,
                output_nonlinearity=tf.nn.softmax,
                name="prob_network"
            )

            self.prob_network = prob_network
            self.feature_network = feature_network
            self.l_input = l_input
            self.state_include_action = state_include_action

            flat_input_var = tf.placeholder(dtype=tf.float32, shape=(None, input_dim), name="flat_input")
            if feature_network is None:
                feature_var = flat_input_var
            else:
                feature_var = L.get_output(l_flat_feature, {feature_network.input_layer: flat_input_var})

            self.f_step_prob = tensor_utils.compile_function(
                [
                    flat_input_var,
                    prob_network.step_prev_hidden_layer.input_var
                ],
                L.get_output([
                    prob_network.step_output_layer,
                    prob_network.step_hidden_layer
                ], {prob_network.step_input_layer: feature_var})
            )

            self.input_dim = input_dim
            self.action_dim = action_dim
            self.hidden_dim = hidden_dim

            self.prev_actions = None
            self.prev_hiddens = None
            self.dist = RecurrentCategorical(env_spec.action_space.n)

            out_layers = [prob_network.output_layer]
            if feature_network is not None:
                out_layers.append(feature_network.output_layer)

            LayersPowered.__init__(self, out_layers)

    @overrides
    def dist_info_sym(self, obs_var, state_info_vars):
        n_batches = tf.shape(obs_var)[0]
        n_steps = tf.shape(obs_var)[1]
        obs_var = tf.reshape(obs_var, tf.pack([n_batches, n_steps, -1]))
        obs_var = tf.cast(obs_var, tf.float32)
        if self.state_include_action:
            prev_action_var = tf.cast(state_info_vars["prev_action"], tf.float32)
            all_input_var = tf.concat(2, [obs_var, prev_action_var])
        else:
            all_input_var = obs_var
        if self.feature_network is None:
            return dict(
                prob=L.get_output(
                    self.prob_network.output_layer,
                    {self.l_input: all_input_var}
                )
            )
        else:
            flat_input_var = tf.reshape(all_input_var, (-1, self.input_dim))
            return dict(
                prob=L.get_output(
                    self.prob_network.output_layer,
                    {self.l_input: all_input_var, self.feature_network.input_layer: flat_input_var}
                )
            )

    @property
    def vectorized(self):
        return True

    def reset(self, dones=None):
        if dones is None:
            dones = [True]
        dones = np.asarray(dones)
        if self.prev_actions is None or len(dones) != len(self.prev_actions):
            self.prev_actions = np.zeros((len(dones), self.action_space.flat_dim))
            self.prev_hiddens = np.zeros((len(dones), self.hidden_dim))

        self.prev_actions[dones] = 0.
        self.prev_hiddens[dones] = self.prob_network.hid_init_param.eval()  # get_value()

    # The return value is a pair. The first item is a matrix (N, A), where each
    # entry corresponds to the action value taken. The second item is a vector
    # of length N, where each entry is the density value for that action, under
    # the current policy
    @overrides
    def get_action(self, observation):
        actions, agent_infos = self.get_actions([observation])
        return actions[0], {k: v[0] for k, v in agent_infos.iteritems()}

    @overrides
    def get_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)
        if self.state_include_action:
            assert self.prev_actions is not None
            all_input = np.concatenate([
                flat_obs,
                self.prev_actions
            ], axis=-1)
        else:
            all_input = flat_obs
        probs, hidden_vec = self.f_step_prob(all_input, self.prev_hiddens)
        actions = special.weighted_sample_n(probs, np.arange(self.action_space.n))
        prev_actions = self.prev_actions
        self.prev_actions = self.action_space.flatten_n(actions)
        self.prev_hiddens = hidden_vec
        agent_info = dict(prob=probs)
        if self.state_include_action:
            agent_info["prev_action"] = np.copy(prev_actions)
        return actions, agent_info

    @property
    @overrides
    def recurrent(self):
        return True

    @property
    def distribution(self):
        return self.dist

    @property
    def state_info_specs(self):
        if self.state_include_action:
            return [
                ("prev_action", (self.action_dim,)),
            ]
        else:
            return []


class EncoderDecoderPolicy(StochasticPolicy, LayersPowered, Serializable):

    def __init__(
            self,
            env_spec,
            num_timesteps,
            vocab_size,
            hidden_sizes=(64,),
            embedding_size=32,
            hidden_nonlinearity=tf.tanh):
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
                               grad_clipping=1.0, only_return_final=True,
                               name="enc_gru")

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
                                    num_timesteps, l_emb, l_dec_out,
                                    grad_clipping=1.0)

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
        probs = theano.printing.Print("probs")(probs)

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


# class RecurrentConversationAgentPolicy(StochasticPolicy, LasagnePowered, Serializable):
#     """
#     Recurrent policy which can either take an action or generate a token
#     sequence at each timestep.
#     """

#     def __init__(
#             self,
#             env_spec,
#             b_agent,
#             hidden_sizes=(32,),
#             embedding_size=32):
#         """
#         Args:
#             env_spec: A spec for the env.
#             hidden_sizes: list of sizes for the fully connected hidden layers
#             hidden_nonlinearity: nonlinearity used for each hidden layer
#         """
#         assert isinstance(env_spec.action_space, Product)
#         assert isinstance(env_spec.action_space.components[0], Discrete)
#         assert isinstance(env_spec.action_space.components[1], DiscreteSequence)

#         Serializable.quick_init(self, locals())
#         super(RecurrentConversationAgentPolicy, self).__init__(env_spec)

#         # TODO remove this requirement
#         assert len(hidden_sizes) == 1

#         # TODO
#         input_shape = (env_spec.observation_space.flat_dim,)

#         # Build input GRU network with embedding lookup.
#         l_inp = L.InputLayer((None, b_agent.num_tokens, b_agent.vocab_size),
#                              input_var=T.itensor3())
#         l_emb = L.EmbeddingLayer(l_inp, input_size=b_agent.vocab_size,
#                                  output_size=embedding_size)
#         l_enc_hid = L.GRULayer(l_emb, num_units=hidden_sizes[0],
#                                only_return_final=True)

#         # Build output GRU network with embedding lookup.
#         # This network is run every timestep when the actor chooses to make an
#         # utterance.
#         # TODO also take in environment state
#         l_dec_out = L.DenseLayer(L.InputLayer((None, hidden_sizes[0])),
#                                  num_units=b_agent.vocab_size,
#                                  nonlinearity=NL.softmax, name="dec_out")
#         l_dec = GRUDecoderLayer(l_enc_hid, hidden_sizes[0], b_agent.num_tokens,
#                                 l_emb, l_dec_out)

#         self._encoder_input = l_inp
#         self._decoder_output = l_dec

#         self._f_prob = ext.compile_function([l_inp.input_var],
#                                             L.get_output([l_dec]))

#         self._prev_action = None
#         self._prev_hidden = None
#         self._hidden_sizes = hidden_sizes
#         #self._dist = RecurrentCategorical(env_spec.action_space.n)

#         self.reset()

#         LasagnePowered.__init__(self, [self._decoder_output])

#     @overrides
#     def dist_info_sym(self, obs_var, state_info_vars):
#         n_batches, n_steps = obs_var.shape[:2]
#         obs_var = obs_var.reshape((n_batches, n_steps, -1))
#         all_input_var = obs_var
#         return dict(
#             p_utterance=L.get_output(
#                 self._decoder_output,
#                 {self._encoder_input: all_input_var}
#             )
#         )

#     def reset(self):
#         # TODO
#         pass

#     # The return value is a pair. The first item is a matrix (N, A), where each
#     # entry corresponds to the action value taken. The second item is a vector
#     # of length N, where each entry is the density value for that action, under
#     # the current policy
#     @overrides
#     def get_action(self, observation):
#         all_input = self.observation_space.flatten(observation)
#         out_probs = self._f_prob([all_input])
#         actions = [special.weighted_sample(out_probs_t, xrange(self.b_agent.vocab_size))
#                    for out_probs_t in out_probs]
#         agent_info = dict(p_utterance=out_probs)
#         return actions, agent_info

#     @property
#     @overrides
#     def recurrent(self):
#         return True

#     @property
#     def distribution(self):
#         raise ValueError("called")
#         return self._dist

#     @property
#     def state_info_keys(self):
#         return []
