"""
Learn a character "bag"-autoencoder which maps from bags of input tokens to
an output sequence. Any output sequence which is some ordering of the bag is
a valid output.
"""

import copy

import tensorflow as tf

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab import config
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import VariantGenerator, stub, run_experiment_lite
from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.core import layers as L
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp

from praglang.environments import BagAutoencoderEnvironment
from praglang.policies import RecurrentCategoricalPolicy
from praglang.util import MLPNetworkWithEmbeddings


stub(globals())

LENGTH = 5
VOCAB = list("abcdefghijklmnopqrstuvwxyz")

env = normalize(BagAutoencoderEnvironment(VOCAB, LENGTH, "autoenc"))


DEFAULTS = {
    "batch_size": 5000,
    "n_itr": 50,
    "step_size": 0.01,
    "policy_hidden_dim": 128,
    "embedding_dim": 32,
    "feature_dim": 128,
    "feature_hidden_dims": (),
}

config.LOG_DIR = "./log"

def run_experiment(params):
    params_base = copy.copy(DEFAULTS)
    params_base.update(params)
    params = params_base

    policy = RecurrentCategoricalPolicy(
            name="policy",
            env_spec=env.spec,
            hidden_dim=params["policy_hidden_dim"],
            feature_network=MLPNetworkWithEmbeddings(
                "embeddings", len(VOCAB), params["feature_dim"],
                params["feature_hidden_dims"], tf.tanh, tf.tanh, len(VOCAB),
                params["embedding_dim"], has_other_input=False),
            state_include_action=False,
    )

    baseline = LinearFeatureBaseline(env.spec)

    optimizer = ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))

    algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=params["batch_size"],
            max_path_length=LENGTH,
            n_itr=params["n_itr"],
            discount=0.99,
            step_size=params["step_size"],
            optimizer=optimizer,
    )

    run_experiment_lite(
            algo.train(),
            n_parallel=5,
            snapshot_mode="last",
            exp_prefix="autoenc_unnorm_reward",
            variant=params,
    )


if __name__ == "__main__":
    run_experiment({})
