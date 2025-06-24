import os

import jax
import jax.numpy as jnp
import numpy as np
from brax.v1.io import html


def run_single_rollout(env, policy_network, params, key, 
                       damage_joint_idx=None | list, 
                       damage_joint_action=None | list,
                       output_dir=None | str):

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
