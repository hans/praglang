from rllab.algos.trpo import TRPO
from rllab.algos.vpg import VPG
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.policies.categorical_gru_policy import CategoricalGRUPolicy

from environments import WordEmissionEnvironment


stub(globals())

env = normalize(WordEmissionEnvironment("wordemit"), normalize_reward=True)

policy = CategoricalGRUPolicy(
        env_spec=env.spec,
        hidden_sizes=(128,),
        state_include_action=False,
)

baseline = LinearFeatureBaseline(env_spec=env.spec)


algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=4000,
        max_path_length=5,
        n_itr=50,
        discount=0.99,
        step_size=0.01,
)


run_experiment_lite(
        algo.train(),
        n_parallel=5,
        snapshot_mode="last",
        log_dir="./log",
)
