from lasagne import init as LI
from lasagne import layers as L
from lasagne import nonlinearities as NL
import theano

from rllab.misc import ext


class GRUStepLayer(L.MergeLayer):
    """
    A gated recurrent unit implements the following update mechanism:
    Reset gate:        r(t) = f_r(x(t) @ W_xr + h(t-1) @ W_hr + b_r)
    Update gate:       u(t) = f_u(x(t) @ W_xu + h(t-1) @ W_hu + b_u)
    Cell gate:         c(t) = f_c(x(t) @ W_xc + r(t) * (h(t-1) @ W_hc) + b_c)
    New hidden state:  h(t) = (1 - u(t)) * h(t-1) + u_t * c(t)
    Note that the reset, update, and cell vectors must have the same dimension as the hidden state
    """

    def __init__(self, incomings,
                 hidden_nonlinearity=NL.tanh,
                 gate_nonlinearity=NL.sigmoid, name=None,
                 W_init=LI.HeUniform(), b_init=LI.Constant(0.)):

        if hidden_nonlinearity is None:
            hidden_nonlinearity = LN.identity

        if gate_nonlinearity is None:
            gate_nonlinearity = LN.identity

        if len(incomings) > 2:
            # Multiple input values.
            all_inputs, hid_prev = incomings[:-1], incomings[-1]
            input_layer = L.ConcatLayer(all_inputs, axis=-1,
                                        name="concat_in_gru_step")
            incomings = [input_layer, hid_prev]

        super(GRUStepLayer, self).__init__(incomings, name=name)

        input_dim = incomings[0].output_shape[-1]
        num_units = incomings[1].output_shape[-1]

        # Weights for the reset gate
        self.W_xr = self.add_param(W_init, (input_dim, num_units), name="W_xr")
        self.W_hr = self.add_param(W_init, (num_units, num_units), name="W_hr")
        self.b_r = self.add_param(b_init, (num_units,), name="b_r", regularizable=False)
        # Weights for the update gate
        self.W_xu = self.add_param(W_init, (input_dim, num_units), name="W_xu")
        self.W_hu = self.add_param(W_init, (num_units, num_units), name="W_hu")
        self.b_u = self.add_param(b_init, (num_units,), name="b_u", regularizable=False)
        # Weights for the cell gate
        self.W_xc = self.add_param(W_init, (input_dim, num_units), name="W_xc")
        self.W_hc = self.add_param(W_init, (num_units, num_units), name="W_hc")
        self.b_c = self.add_param(b_init, (num_units,), name="b_c", regularizable=False)
        self.gate_nonlinearity = gate_nonlinearity
        self.num_units = num_units
        self.nonlinearity = hidden_nonlinearity

    def get_output_for(self, inputs, **kwargs):
        x, hprev = inputs

        r = self.gate_nonlinearity(x.dot(self.W_xr) + hprev.dot(self.W_hr) + self.b_r)
        u = self.gate_nonlinearity(x.dot(self.W_xu) + hprev.dot(self.W_hu) + self.b_u)
        c = self.nonlinearity(x.dot(self.W_xc) + r * (hprev.dot(self.W_hc)) + self.b_c)
        h = (1 - u) * hprev + u * c
        return h.astype(theano.config.floatX)

    def get_output_shape_for(self, input_shape):
        return input_shape[0], self.num_units
