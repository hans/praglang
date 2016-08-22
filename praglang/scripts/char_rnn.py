"""
Learn a recurrent policy which outputs valid English character sequences.
The environment rewards the agent for outputting valid prefixes, and the
agent quickly learns a simple character-level English language model.

Unfortunately the entropy of the action space is rather low. After ~20
iterations policy outputs ~170 unique words; only 30-40 of them are valid.
"""


from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp

from praglang.environments import WordEmissionEnvironment
from praglang.policies import RecurrentCategoricalPolicy


stub(globals())

env = normalize(WordEmissionEnvironment("wordemit"), normalize_reward=True)

policy = RecurrentCategoricalPolicy(
        name="policy",
        env_spec=env.spec,
        hidden_dim=128,
        state_include_action=False,
        #temperature=2,
)

baseline = LinearFeatureBaseline(env_spec=env.spec)


algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=50000,
        max_path_length=5,
        n_itr=50,
        discount=0.99,
        step_size=0.01,
        optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
)


run_experiment_lite(
        algo.train(),
        n_parallel=5,
        snapshot_mode="last",
        log_dir="./log",
)
