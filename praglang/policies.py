import numpy as np
import tensorflow as tf

from rllab.core.serializable import Serializable
from rllab.misc import special
from rllab.misc.overrides import overrides
from sandbox.rocky.tf.core import layers as L
from sandbox.rocky.tf.core.layers_powered import LayersPowered
from sandbox.rocky.tf.distributions.recurrent_categorical import RecurrentCategorical
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.policies.base import StochasticPolicy
from sandbox.rocky.tf.spaces.discrete import Discrete
from sandbox.rocky.tf.spaces.product import Product

from praglang.spaces import DiscreteSequence
from praglang.util import DeepGRUNetwork


class RecurrentCategoricalPolicy(StochasticPolicy, LayersPowered, Serializable):

    def __init__(
            self,
            name,
            env_spec,
            hidden_dims=(32,),
            feature_network=None,
            state_include_action=True,
            hidden_nonlinearity=tf.tanh):
        """
        :param env_spec: A spec for the env.
        :param hidden_dims: dimension of hidden layers
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

            prob_network = DeepGRUNetwork(
                input_shape=(feature_dim,),
                input_layer=l_feature,
                output_dim=env_spec.action_space.n,
                hidden_dims=hidden_dims,
                hidden_nonlinearity=hidden_nonlinearity,
                output_nonlinearity=tf.nn.softmax,
                name="prob_network"
            )

            self.prob_network = prob_network
            self.feature_network = feature_network
            self.l_input = l_input
            self.state_include_action = state_include_action

            flat_input_var = tf.placeholder(tf.float32, shape=(None, input_dim), name="flat_input")
            if feature_network is None:
                feature_var = flat_input_var
            else:
                feature_var = L.get_output(l_flat_feature, {feature_network.input_layer: flat_input_var})

            # Build the step feedforward function.
            inputs = [flat_input_var] \
                    + [prev_hidden.input_var for prev_hidden
                            in prob_network.step_prev_hidden_layers]
            outputs = [prob_network.step_output_layer] \
                    + prob_network.step_hidden_layers
            outputs = L.get_output(outputs, {prob_network.step_input_layer: feature_var})
            self.f_step_prob = tensor_utils.compile_function(
                    inputs, outputs)

            # Function to fetch hidden init values
            self.f_hid_inits = tensor_utils.compile_function(
                    [], prob_network.hid_inits)

            self.input_dim = input_dim
            self.action_dim = action_dim
            self.hidden_dims = hidden_dims

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
            self.prev_hiddens = [np.zeros((len(dones), hidden_dim))
                                 for hidden_dim in self.hidden_dims]

        self.prev_actions[dones] = 0.

        prev_hiddens = self.f_hid_inits()
        for prev_hidden, tgt in zip(prev_hiddens, self.prev_hiddens):
            # Broadcast.
            tgt[dones] = prev_hidden

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

        ret = self.f_step_prob(all_input, *self.prev_hiddens)
        probs, hidden_vecs = ret[0], ret[1:]

        actions = special.weighted_sample_n(probs, np.arange(self.action_space.n))
        prev_actions = self.prev_actions

        self.prev_actions = self.action_space.flatten_n(actions)
        self.prev_hiddens = hidden_vecs

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
