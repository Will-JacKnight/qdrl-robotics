from typing import Optional
import os

import jax
import jax.numpy as jnp
import numpy as np
from brax.v1.io import html

import qdax.tasks.brax.v1 as environments
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.tasks.brax.v1.wrappers.reward_wrappers import OffsetRewardWrapper, ClipRewardWrapper

from utils.reward_wrapper import WalkRewardOnlyWrapper


def init_env_and_policy_network(env_name, episode_length, policy_hidden_layer_sizes):
    """
    init environment and policy network for adaptation tests and evaluation use
    """
    # Init brax environment
    env = environments.create(env_name, episode_length=episode_length)
    # env = OffsetRewardWrapper(env, offset=environments.reward_offset[env_name])
    # env = ClipRewardWrapper(env, clip_min=0.,)
    env = WalkRewardOnlyWrapper(env, env_name)

    # Init policy network
    policy_layer_sizes = policy_hidden_layer_sizes + (env.action_size,)
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
            )
    return env, policy_network


def run_single_rollout(env, policy_network, params, key, 
                       damage_joint_idx: Optional[list] = None, 
                       damage_joint_action: Optional[list] = None,
                       output_dir: Optional[str] = None):

    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)
    jit_inference_fn = jax.jit(policy_network.apply)

    states = []
    rewards = []
    actions = []

    key, subkey = jax.random.split(key)
    state = jit_env_reset(rng=subkey)

    while not state.done:
        states.append(state)
        action = jit_inference_fn(params, state.obs)
        if damage_joint_idx is not None:
            action = action.at[jnp.array(damage_joint_idx)].set(damage_joint_action)
        actions.append(action)
        state = jit_env_step(state, action)     # get next state
        rewards.append(state.reward)

    if (output_dir is not None):
        with open(output_dir, "w") as f:
            f.write(html.render(env.sys, [s.qp for s in states[:500]]))
            print("Animation generated.")

    return {
        'rewards': np.array(rewards),
        'states': np.array(states),
        'actions': np.array(actions)
        }
