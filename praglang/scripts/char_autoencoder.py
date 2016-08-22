"""
Learn a character "bag"-autoencoder which maps from bags of input tokens to
an output sequence. Any output sequence which is some ordering of the bag is
a valid output.
"""


from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp

from praglang.baselines import MovingAverageBaseline
from praglang.environments import BagAutoencoderEnvironment
from praglang.policies import RecurrentCategoricalPolicy


stub(globals())

LENGTH = 3
VOCAB = list("abcde")

env = normalize(BagAutoencoderEnvironment(VOCAB, LENGTH, "autoenc"), normalize_reward=True)

policy = RecurrentCategoricalPolicy(
        name="policy",
        env_spec=env.spec,
        hidden_dim=128,
        state_include_action=False,
)

#baseline = MovingAverageBaseline(env.spec)
baseline = ZeroBaseline(env.spec)
#baseline = LinearFeatureBaseline(env.spec)

optimizer = ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))

algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=50000,#100,#5000,#50000,
        max_path_length=LENGTH,
        n_itr=50,
        discount=0.99,
        step_size=0.01,
        optimizer=optimizer,
)


run_experiment_lite(
        algo.train(),
        n_parallel=1,#5,
        snapshot_mode="last",
        log_dir="./log/autoenc",
)
