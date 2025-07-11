from typing import Optional, Dict
import os

import jax
import jax.numpy as jnp
import numpy as np
from brax.v1.io import html

import qdax.tasks.brax.v1 as environments
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.tasks.brax.v1.wrappers.reward_wrappers import OffsetRewardWrapper, ClipRewardWrapper

from utils.reward_wrapper import ForwardStepRewardWrapper
from utils.util import load_pkls
from utils.new_plot import plot_grid_results


def init_env_and_policy_network(env_name, episode_length, policy_hidden_layer_sizes):
    """
    init environment and policy network for adaptation tests and evaluation use
    """
    # Init brax environment
    env = environments.create(env_name, episode_length=episode_length)
    # env = OffsetRewardWrapper(env, offset=environments.reward_offset[env_name])
    # env = ClipRewardWrapper(env, clip_min=0.,)
    env = ForwardStepRewardWrapper(env, env_name)

    # Init policy network
    policy_layer_sizes = policy_hidden_layer_sizes + (env.action_size,)
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
            )
    return env, policy_network


def render_rollout_to_html(states, env, output_path):
    with open(output_path, "w") as f:
        f.write(html.render(env.sys, [s.qp for s in states]))
        print("Animation generated.")


def run_single_rollout(
    env, 
    policy_network, 
    params, 
    key, 
    damage_joint_idx: Optional[jnp.ndarray] = None, 
    damage_joint_action: Optional[jnp.ndarray] = None,
) -> Dict:
    """
    non-jittable version for single rollout
    """
    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)
    jit_inference_fn = jax.jit(policy_network.apply)

    states = []
    rewards = []
    actions = []
    # r_survive = []
    # r_forward = []
    # r_control = []
    # r_contact = []

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
        # r_survive.append(state.metrics["reward_survive"])
        # r_forward.append(state.metrics["reward_forward"])
        # r_control.append(state.metrics["reward_ctrl"])
        # r_contact.append(state.metrics["reward_contact"])

    return {
        'rewards': np.array(rewards),
        'states': np.array(states),
        'actions': np.array(actions),
        # 'survive_reward': np.array(r_survive),
        # 'forward_reward': np.array(r_forward),
        # 'control_reward': np.array(r_control),
        # 'contact_reward': np.array(r_contact),
        }

def create_jit_rollout_fn(env, policy_network, episode_length: int):
    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)
    jit_inference_fn = jax.jit(policy_network.apply)
    
    @jax.jit
    def _jit_run_single_rollout(
        params, 
        key, 
        damage_joint_idx: jnp.ndarray, 
        damage_joint_action: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        jit version for single rollout, on the assumption that 
            - the rollout always lasts episode_length steps and there're no early terminations
            - damage is always applied

        returns:
            - total_rewards: fitness of the rollout, calculated as the sum of step rewards
        """

        key, subkey = jax.random.split(key)
        state = jit_env_reset(rng=subkey)

        def step_fn(carry, _):
            state, total_rewards = carry
            action = jit_inference_fn(params, state.obs)
            action = action.at[damage_joint_idx].set(damage_joint_action)
            state = jit_env_step(state, action)     # get next state
            reward = state.reward
            carry = (state, total_rewards + reward)
            return carry, reward
        
        init_carry = (state, jnp.float32(0.0))
        (_, total_rewards), _ = jax.lax.scan(step_fn, init_carry, length=episode_length)

        return total_rewards

    return _jit_run_single_rollout


if __name__ == "__main__":
    key = jax.random.key(42)
    # output_path = "./outputs/dcrl_20250703_114735"
    output_path = "./outputs/mapelites_20250701_152736"
    damage_joint_idx = jnp.array([0,1])
    damage_joint_action = jnp.array([0,0.9])


    repertoire, _ = load_pkls(output_path)
    env, policy_network = init_env_and_policy_network("ant_uni", 1000, (32,32))

    jit_rollout_fn = create_jit_rollout_fn(env, policy_network, 1000)

    def single_eval(param, key):
        return jit_rollout_fn(param, key, damage_joint_idx, damage_joint_action)

    while True:
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, 10000)
        batched_rewards = jax.vmap(single_eval)(repertoire.genotypes, keys)
        repertoire = repertoire.replace(fitnesses=batched_rewards.reshape((-1, 1)))
        plot_grid_results("real", repertoire, jnp.array(0.), jnp.array(1.), (10,10,10,10), output_path)

        best_real_idx = jnp.argmax(batched_rewards)
        best_params = jax.tree.map(lambda x: x[best_real_idx], repertoire.genotypes)
        key, subkey = jax.random.split(key)
        rollout = run_single_rollout(env, policy_network, best_params, subkey, damage_joint_idx, damage_joint_action)
        render_rollout_to_html(rollout['states'], env, output_path + "/best_real_fitness.html")
        breakpoint()