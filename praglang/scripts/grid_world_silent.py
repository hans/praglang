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

import copy

import numpy as np
import tensorflow as tf

from rllab import config
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.core.network import MLP
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy

from praglang.agents.grid import GridWorldMasterAgent, GridWorldSlaveAgent
from praglang.environments.grid import GridWorldEnv, SlaveGridWorldEnv
from praglang.environments.conversation import SituatedConversationEnvironment


stub(globals())

DEFAULTS = {
    "batch_size": 5000,
    "n_itr": 500,
    "step_size": 0.01,
    "policy_hidden_dims": (32,32),

    "goal_reward": 10.0, # reward for reaching goal
}

config.LOG_DIR = "./log"

def run_experiment(**params):
    base_params = copy.copy(DEFAULTS)
    base_params.update(params)
    params = base_params

    grid_world = SlaveGridWorldEnv("3x3", goal_reward=params["goal_reward"])
    env = normalize(grid_world)
    baseline = LinearFeatureBaseline(env)

    policy = CategoricalMLPPolicy(
            name="policy",
            env_spec=env.spec,
            hidden_sizes=params["policy_hidden_dims"],
    )

    optimizer = ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))

    algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=params["batch_size"],
            max_path_length=5,
            n_itr=params["n_itr"],
            discount=0.99,
            step_size=params["step_size"],
            optimizer=optimizer,
    )

    run_experiment_lite(
            algo.train(),
            n_parallel=5,
            snapshot_mode="last",
            exp_prefix="grid_world_silent",
            variant=params,
    )


run_experiment()
