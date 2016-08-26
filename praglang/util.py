import itertools

import numpy as np
import tensorflow as tf

from rllab.core.serializable import Serializable
from sandbox.rocky.tf.core import layers as L
from sandbox.rocky.tf.core.layers_powered import LayersPowered


def find_placeholder_ancestors(tensor):
    queue = [tensor]
    ret = set()
    while len(queue):
        x = queue.pop()

        ret = ret | set([y for y in x.op.inputs if y.op.type == "Placeholder"])
        queue.extend(list(x.op.inputs))

    return ret


def uniform_init(shape, dtype=tf.float32):
    return tf.random_uniform(shape=shape)


class MeanPoolEmbeddingLayer(L.Layer):
    """
    Mean-pool embedding lookup.

    Very inefficient for large vocabularies.
    """

    def __init__(self, incoming, name, output_dim,
                 W_emb=tf.random_uniform_initializer(),
                 **kwargs):
        super(MeanPoolEmbeddingLayer, self).__init__(incoming, name, **kwargs)

        self.output_dim = output_dim
        input_dim = int(np.prod(self.input_shape[1:]))

        with tf.variable_scope(name):
            self.W_emb = self.add_param(W_emb, (input_dim, output_dim), name="W_emb")

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_output_for(self, input, **kwargs):
        # Mean-pool denominator
        scale = tf.reduce_sum(input, -1, keep_dims=True) + 1e-5

        activation = tf.matmul(input, self.W_emb)
        activation /= scale

        return activation


class MLPNetworkWithEmbeddings(LayersPowered, Serializable):

    """
    An MLP which takes some embeddings at the input layer. These should be
    initialized differently than the rest of the parameters, so we manage that
    in this custom network.
    """

    def __init__(self, name, input_dim, output_dim, hidden_sizes,
                 hidden_nonlinearity, output_nonlinearity, vocab_size,
                 embedding_size, hidden_W_init=L.xavier_init,
                 hidden_b_init=tf.zeros_initializer,
                 output_W_init=L.xavier_init,
                 output_b_init=tf.zeros_initializer,
                 has_other_input=True, input_var=None, input_layer=None,
                 **kwargs):
        Serializable.quick_init(self, locals())

        with tf.variable_scope(name):
            if input_layer is None:
                input_layer = L.InputLayer(shape=(None, input_dim),
                                           input_var=input_var, name="input")
            l_in = input_layer

            if has_other_input:
                # Slice apart
                l_other_in = L.SliceLayer(l_in, "slice_other",
                                        slice(0, input_dim - vocab_size),
                                        axis=-1)
                l_emb_in = L.SliceLayer(l_in, "slice_emb",
                                        slice(input_dim - vocab_size, input_dim),
                                        axis=-1)

                # HACK: This is cheap with small embedding matrices but will not scale well..
                # Find a better way to lookup from this representation + mean-pool
                l_embs = MeanPoolEmbeddingLayer(l_emb_in, "embeddings", embedding_size)

                l_hidden_input = L.ConcatLayer([l_other_in, l_embs], "merge")
            else:
                l_hidden_input = l_in

            hidden_layers = [l_hidden_input]
            for i, hidden_size in enumerate(hidden_sizes):
                l_hid = L.DenseLayer(hidden_layers[-1], num_units=hidden_size,
                                     nonlinearity=hidden_nonlinearity,
                                     name="hidden_%i" % i,
                                     W=hidden_W_init, b=hidden_b_init)
                hidden_layers.append(l_hid)

            l_out = L.DenseLayer(hidden_layers[-1], num_units=output_dim,
                                 nonlinearity=output_nonlinearity,
                                 name="output",
                                 W=output_W_init, b=output_b_init)

            self.input_layer = l_in
            self.input_var = l_in.input_var
            self.output_layer = l_out

            LayersPowered.__init__(self, l_out)


class DeepGRUStepLayer(L.MergeLayer):

    """
    Run a single step of a deep GRU.
    """

    def __init__(self, l_step_input, l_step_prev_hiddens, l_grus, name="deep_gru_step"):
        assert len(l_step_prev_hiddens) == len(l_grus)

        incomings = [l_step_input] + l_step_prev_hiddens
        super(DeepGRUStepLayer, self).__init__(incomings, name)
        self._l_grus = l_grus

    def get_params(self, **tags):
        ret = itertools.chain.from_iterable(l_gru.get_params(**tags) for l_gru in self._l_grus)

    def get_output_shape_for(self, input_shapes):
        n_batch = input_shapes[0]
        return n_batch, self._l_grus[-1].num_units

    def get_output_for(self, inputs, **kwargs):
        x = inputs[0]
        hprevs = inputs[1:]

        n_batch = tf.shape(x)[0]
        x = tf.reshape(x, tf.pack((n_batch, -1)))

        input_val = x
        for hprev, l_gru in zip(hprevs, self._l_grus):
            input_val = l_gru.step(hprev, input_val)

        return input_val


class DeepGRUNetwork(object):
    """
    Multilayer extension of `rllab.core.network.GRUNetwork`
    """

    def __init__(self, name, input_shape, output_dim, hidden_dims, hidden_nonlinearity=tf.nn.relu,
                 output_nonlinearity=None, input_var=None, input_layer=None):
        with tf.variable_scope(name):
            if input_layer is None:
                l_in = L.InputLayer(shape=(None, None) + input_shape, input_var=input_var, name="input")
            else:
                l_in = input_layer

            l_step_input = L.InputLayer(shape=(None,) + input_shape, name="step_input")
            l_step_prev_hiddens = [L.InputLayer(shape=(None, hidden_dim), name="step_prev_hidden%i" % i)
                                   for i, hidden_dim in enumerate(hidden_dims)]

            # Build the unrolled GRU network, which operates laterally, then
            # vertically
            below = l_in
            l_grus = []
            for i, hidden_dim in enumerate(hidden_dims):
                l_gru = L.GRULayer(below, num_units=hidden_dim, hidden_nonlinearity=hidden_nonlinearity,
                                   hidden_init_trainable=False, name="gru%i" % i)
                l_grus.append(l_gru)
                below = l_gru

            # Convert final hidden layer to flat representation
            l_gru_flat = L.ReshapeLayer(
                l_grus[-1], shape=(-1, hidden_dims[-1]),
                name="gru_flat"
            )
            l_output_flat = L.DenseLayer(
                l_gru_flat,
                num_units=output_dim,
                nonlinearity=output_nonlinearity,
                name="output_flat"
            )
            l_output = L.OpLayer(
                l_output_flat,
                op=lambda flat_output, l_input:
                tf.reshape(flat_output, tf.pack((tf.shape(l_input)[0], tf.shape(l_input)[1], -1))),
                shape_op=lambda flat_output_shape, l_input_shape:
                (l_input_shape[0], l_input_shape[1], flat_output_shape[-1]),
                extras=[l_in],
                name="output"
            )

            # Build a single step of the GRU network, which operates vertically
            # and is replicated laterally
            below = l_step_input
            l_step_hiddens = []
            for i, (l_gru, prev_hidden) in enumerate(zip(l_grus, l_step_prev_hiddens)):
                l_step_hidden = L.GRUStepLayer([below, prev_hidden],
                                               "step_hidden%i" % i,
                                               l_gru)
                l_step_hiddens.append(l_step_hidden)
                below = l_step_hidden

            l_step_output = L.DenseLayer(
                l_step_hiddens[-1],
                num_units=output_dim,
                nonlinearity=output_nonlinearity,
                W=l_output_flat.W,
                b=l_output_flat.b,
                name="step_output"
            )

            self._l_in = l_in
            self._hid_inits = [l_gru.h0 for l_gru in l_grus]
            self._l_grus = l_grus
            self._l_out = l_output

            self._l_step_input = l_step_input
            self._l_step_prev_hiddens = l_step_prev_hiddens
            self._l_step_hiddens = l_step_hiddens
            self._l_step_output = l_step_output

    @property
    def input_layer(self):
        return self._l_in

    @property
    def input_var(self):
        return self._l_in.input_var

    @property
    def output_layer(self):
        return self._l_out

    @property
    def step_input_layer(self):
        return self._l_step_input

    @property
    def step_prev_hidden_layers(self):
        return self._l_step_prev_hiddens

    @property
    def step_hidden_layers(self):
        return self._l_step_hiddens

    @property
    def step_output_layer(self):
        return self._l_step_output

    @property
    def hid_inits(self):
        return self._hid_inits

