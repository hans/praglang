"""
A conversational agent learns to navigate a grid world with the help of a
conversational-agent friend.

We learn a recurrent policy `A` which can take actions in a grid-world
environment and send messages (token sequences) to a friend `B`. `A` is
*blind*, while `B` has full vision of `A`'s grid world environment. `A` must
communicate with `B` in order to find its goal and then reach it.

Here `B` is a simple fixed agent which runs regex matches on `A`'s utterances.
Its language is vaguely English-like, but aggressively abbreviated in order to
make `A`'s task easier.
"""

from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.envs.grid_world_env import GridWorldEnv
from rllab.misc.instrument import stub, run_experiment_lite

from praglang.agents.grid import GridWorldAgent
from praglang.environments.grid import GridWorldEnv
from praglang.environments.conversation import SituatedConversationEnvironment
from praglang.policies import RecurrentConversationAgentPolicy


stub(globals())

agent = GridWorldAgent()

base_env = GridWorldEnv("3x3")
env = normalize(SituatedConversationEnvironment(base_env, agent))
baseline = LinearFeatureBaseline(env)

policy = RecurrentConversationAgentPolicy(
        env_spec=env.spec,
        b_agent=agent,
        hidden_sizes=(128,),
)

algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=50000,
        max_path_length=10,
        n_itr=50,
        discount=0.99,
        step_size=0.01,
)

run_experiment_lite(
        algo.train(),
        n_parallel=1,
        snapshot_mode="last",
        log_dir="./log/grid_world/log",
)
