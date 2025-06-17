import jax
import jax.numpy as jnp
import numpy as np

from brax.v1.io import html

def run_single_rollout(env, policy_network, params, key, 
                       damage_joint_idx=None, damage_joint_action=0,
                       save_animation=True):

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
            action = action.at[damage_joint_idx].set(damage_joint_action)
        actions.append(action)
        state = jit_env_step(state, action)     # get next state
        rewards.append(state.reward)

    if (save_animation):
        with open("./outputs/brax_rollout.html", "w") as f:
            f.write(html.render(env.sys, [s.qp for s in states[:500]]))
            print("Animation generated.")

    return {
        'rewards': np.array(rewards),
        'states': np.array(states),
        'actions': np.array(actions)
        }

# def collect_rollouts(rollout):
#     rollouts = {}
#     for i in 
#         rollouts[i] = rollout
#     return rollouts