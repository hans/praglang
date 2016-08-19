"""
Learn a recurrent policy which outputs valid English character sequences.
The environment rewards the agent for outputting valid prefixes, and the
agent quickly learns a simple character-level English language model.

Unfortunately the entropy of the action space is rather low. After ~20
iterations policy outputs ~170 unique words; only 30-40 of them are valid.
"""


from rllab.algos.trpo import TRPO
from rllab.algos.vpg import VPG
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite

from praglang.environments import AutoencoderEnvironment
from praglang.policies import EncoderDecoderPolicy


stub(globals())

LENGTH = 3
VOCAB = list("abcde")

autoenc_env = AutoencoderEnvironment(VOCAB, LENGTH, "autoenc")
env = normalize(autoenc_env, normalize_reward=True)

policy = EncoderDecoderPolicy(
        env_spec=env.spec,
        num_timesteps=LENGTH,
        vocab_size=len(VOCAB),
        hidden_sizes=(128,),
)

baseline = ZeroBaseline(env.spec)


algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=100,#5000,#50000,
        max_path_length=LENGTH,
        n_itr=50,
        discount=0.99,
        step_size=0.01,
        debug_nan=True,
)


run_experiment_lite(
        algo.train(),
        n_parallel=1,#5,
        snapshot_mode="last",
        log_dir="./log/autoenc",
)
