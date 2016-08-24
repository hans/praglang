import numpy as np
import tensorflow as tf

from rllab.core.serializable import Serializable
from sandbox.rocky.tf.core import layers as L
from sandbox.rocky.tf.core.layers_powered import LayersPowered


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
        if input.get_shape().ndims > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = tf.reshape(input, tf.pack([tf.shape(input)[0], -1]))

        activation = tf.matmul(input, self.W_emb)
        activation /= tf.reduce_sum(input, 1, keep_dims=True) + 1e-5

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
                                        slice(0, input_dim - vocab_size + 1),
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
