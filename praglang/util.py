import numpy as np
import tensorflow as tf

from sandbox.rocky.tf.core import layers as L


def uniform_init(shape, dtype=tf.float32):
    return tf.random_uniform(shape=shape)


class MeanPoolEmbeddingLayer(L.Layer):
    """
    Cheap mean-pool embedding lookup.

    Very inefficient for large vocabularies.
    """

    def __init__(self, incoming, name, num_units, W=tf.random_uniform_initializer(), **kwargs):
        super(MeanPoolEmbeddingLayer, self).__init__(incoming, name, **kwargs)

        self.num_units = num_units
        num_inputs = int(np.prod(self.input_shape[1:]))

        self.W = self.add_param(W, (num_inputs, num_units), name="W")

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, **kwargs):
        if input.get_shape().ndims > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = tf.reshape(input, tf.pack([tf.shape(input)[0], -1]))

        activation = tf.matmul(input, self.W)
        # TODO won't work with flattened seqs I think?
        activation /= tf.reduce_sum(input, 1, keep_dims=True) + 1e-5

        return activation


class MeanPoolEmbeddingNetwork(object):
    def __init__(self, name, vocab_size, embedding_size, input_var=None,
                 input_layer=None):
        with tf.variable_scope(name):
            embeddings = tf.get_variable("embeddings", (vocab_size, embedding_size))

            if input_layer is None:
                input_layer = L.InputLayer(shape=(None, vocab_size), input_var=input_var, name="input")
            l_in = input_layer

            # HACK: This is cheap with small embedding matrices but will not scale well..
            # Find a better way to lookup from this representation + mean-pool
            l_output = MeanPoolEmbeddingLayer(l_in, "embeddings", embedding_size)

            self.input_layer = l_in
            self.input_var = l_in.input_var
            self.output_layer = l_output


