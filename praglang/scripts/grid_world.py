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

from praglang.agents.grid import GridWorldMasterAgent, GridWorldSlaveAgent
from praglang.environments.grid import GridWorldEnv, SlaveGridWorldEnv
from praglang.environments.conversation import SituatedConversationEnvironment
from praglang.policies import RecurrentCategoricalPolicy
from praglang.util import MLPNetworkWithEmbeddings


stub(globals())

EMBEDDING_DIM = 32

grid_world = SlaveGridWorldEnv("walled_chain")
agent = GridWorldMasterAgent(grid_world)
env = normalize(SituatedConversationEnvironment(env=grid_world, b_agent=agent))
baseline = LinearFeatureBaseline(env)

DEFAULTS = {
    "batch_size": 50000,
    "n_itr": 100,
    "step_size": 0.01,
    "policy_hidden_dims": (128,),
    "embedding_dim": 32,
    "feature_dim": 128,
    "feature_hidden_dims": (128,),
}

config.LOG_DIR = "./log"

def run_experiment(params):
    base_params = copy.copy(DEFAULTS)
    base_params.update(params)
    params = base_params

    policy = RecurrentCategoricalPolicy(
            name="policy",
            env_spec=env.spec,
            hidden_dims=params["policy_hidden_dims"],
            feature_network=MLPNetworkWithEmbeddings(
                "feature_network", env.observation_space.flat_dim,
                params["feature_dim"], params["feature_hidden_dims"],
                tf.tanh, tf.tanh, agent.vocab_size, params["embedding_dim"]),
            state_include_action=False,
    )

    optimizer = ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))

    algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=params["batch_size"],
            max_path_length=10,
            n_itr=params["n_itr"],
            discount=0.99,
            step_size=params["step_size"],
            optimizer=optimizer,
    )

    run_experiment_lite(
            algo.train(),
            n_parallel=5,
            snapshot_mode="last",
            exp_prefix="grid_world",
            variant=params,
    )

run_experiment({})
