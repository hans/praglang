import numpy as np
import theano
from theano import tensor as T

from rllab.baselines.base import Baseline
from rllab.misc.ext import compile_function
from rllab.misc.overrides import overrides

class MovingAverageBaseline(Baseline):
    def __init__(self, env_spec, tau=0.9):
        self._baseline = theano.shared(0.0, name="baseline")

        return_mean = T.scalar("empirical_return_mean")
        updated_baseline = \
                tau * self._baseline \
                  + (1 - tau) * return_mean

        self._update_baseline = compile_function(
            inputs=[return_mean],
            updates={self._baseline: updated_baseline})

    @overrides
    def fit(self, paths):
        self._update_baseline(np.mean([np.mean(path["returns"]) for path in paths]))

    @overrides
    def predict(self, path):
        return self._baseline.get_value() * np.ones_like(path["rewards"])
