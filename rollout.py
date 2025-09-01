from typing import Optional, Dict, Tuple
import os
import functools

import jax
import jax.numpy as jnp
import numpy as np
from brax.v1.io import html

import qdax.tasks.brax.v1 as environments
# from qdax.core.neuroevolution.networks.networks import MLP, MLPDC
from qdax.core.neuroevolution.buffers.buffer import DCRLTransition
from qdax.tasks.brax.v1.wrappers.reward_wrappers import OffsetRewardWrapper, ClipRewardWrapper
from qdax.tasks.brax.v1.wrappers.init_state_wrapper import FixedInitialStateWrapper
from qdax.custom_types import EnvState, Params, RNGKey
from qdax.tasks.brax.v1 import descriptor_extractor
from qdax.tasks.brax.v1.env_creators import scoring_function_brax_envs
from qdax.utils.metrics import default_qd_metrics

from utils.reward_wrapper import ForwardStepRewardWrapper
from utils.networks import ResMLP, ResMLPDC, DropoutMLP, DropoutMLPDC


def setup_environment(
    env_name: str, 
    episode_length: int, 
    policy_hidden_layer_sizes: Tuple[int, ...], 
    dropout_rate: float,
    init_batch_size: int,
    key: RNGKey,
):
    # Init brax environment
    env = environments.create(env_name, episode_length=episode_length)

    wrapper_env_name = "ant" if env_name == "ant_uni" else env_name
    # env = OffsetRewardWrapper(env, offset=environments.reward_offset[env_name])
    # env = ClipRewardWrapper(env, clip_min=0.,)
    env = ForwardStepRewardWrapper(env, wrapper_env_name)
    # env = FixedInitialStateWrapper(env, env_name)

    # Init policy network
    policy_layer_sizes = policy_hidden_layer_sizes + (env.action_size,)
    policy_network = DropoutMLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
        dropout_rate=dropout_rate
    )
    
    actor_dc_network = DropoutMLPDC(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
        dropout_rate=dropout_rate
    )
    
    reset_fn = jax.jit(env.reset)

    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num=init_batch_size)
    fake_batch_obs = jnp.zeros(shape=(init_batch_size, env.observation_size))
    init_params = jax.vmap(policy_network.init)(keys, fake_batch_obs)

    def play_step_fn(
        env_state: EnvState, policy_params: Params, key: RNGKey
    ) -> Tuple[EnvState, Params, RNGKey, DCRLTransition]:
        key, subkey = jax.random.split(key)
        actions = policy_network.apply(policy_params, env_state.obs, train=True, rngs={"dropout": subkey})
        state_desc = env_state.info["state_descriptor"]
        next_state = env.step(env_state, actions)

        transition = DCRLTransition(
            obs=env_state.obs,
            next_obs=next_state.obs,
            rewards=next_state.reward,
            dones=next_state.done,
            truncations=next_state.info["truncation"],
            actions=actions,
            state_desc=state_desc,
            next_state_desc=next_state.info["state_descriptor"],
            desc=jnp.zeros(
                env.descriptor_length,
            )
            * jnp.nan,
            desc_prime=jnp.zeros(
                env.descriptor_length,
            )
            * jnp.nan,
        )

        return next_state, policy_params, key, transition

    # reevaluation for corrected metrics
    descriptor_extraction_fn = descriptor_extractor[env_name]
    scoring_fn = functools.partial(
        scoring_function_brax_envs,
        episode_length=episode_length,
        play_reset_fn=reset_fn,
        play_step_fn=play_step_fn,
        descriptor_extractor=descriptor_extraction_fn,
    )

    reward_offset = environments.reward_offset[env_name]
    metrics_fn = functools.partial(
        default_qd_metrics,
        qd_offset=reward_offset * episode_length,
    )
    return env, policy_network, actor_dc_network, reset_fn, play_step_fn, scoring_fn, metrics_fn, init_params


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
    zero_sensor_idx: Optional[jnp.ndarray] = None
) -> Dict:
    """
    non-jittable version for single rollout (only supports ant_uni).

    args:
        - demage_joint_idx: indices range from [0, 7]
        - damage_joint_action: torque value ranges from [-1, 1]
        - zero_sensor_idx: indices range from [0, 86]
            - [0, 12]: joint positions
            - [13, 26]: joint velocities
            - [27, 86]: p.s. contact forces
            - brax ant doc: https://github.com/google/brax/blob/main/brax/envs/ant.py

    """
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

        # simulate failing sensors
        if zero_sensor_idx is not None:
            damaged_obs = state.obs.at[zero_sensor_idx].set(0.0)
            state = state.replace(obs=damaged_obs)

        key, subkey = jax.random.split(key)
        action = jit_inference_fn(params, state.obs, rngs={"dropout": subkey})

        # apply damage to actions
        if damage_joint_idx is not None:
            action = action.at[damage_joint_idx].set(damage_joint_action)
        actions.append(action)

        # get next state
        state = jit_env_step(state, action)     

        rewards.append(state.reward)

    return {
        'rewards': np.array(rewards),
        'states': np.array(states),
        'actions': np.array(actions),
        }


def jit_rollout_fn(
    env, 
    policy_network, 
    episode_length: int,
):
    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)
    jit_inference_fn = jax.jit(policy_network.apply)
    
    @jax.jit
    def _jit_fitness_rollout(
        params, 
        key, 
        damage_joint_idx: jnp.ndarray, 
        damage_joint_action: jnp.ndarray,
        zero_sensor_idx: jnp.ndarray,
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
            state, total_rewards, key = carry
            damaged_obs = state.obs.at[zero_sensor_idx].set(0.0)
            state = state.replace(obs=damaged_obs)
            key, subkey = jax.random.split(key)
            action = jit_inference_fn(params, state.obs, rngs={"dropout": subkey})
            action = action.at[damage_joint_idx].set(damage_joint_action)
            state = jit_env_step(state, action)     # get next state
            reward = state.reward
            carry = (state, total_rewards + reward, key)
            return carry, reward
        
        init_carry = (state, jnp.float32(0.0), key)
        (_, total_rewards, _), _ = jax.lax.scan(step_fn, init_carry, length=episode_length)

        return total_rewards

    return _jit_fitness_rollout


def play_damage_step_fn(
    env_state: EnvState, 
    policy_params: Params, 
    key: RNGKey,
    env,
    policy_network,
    damage_joint_idx: jnp.ndarray, 
    damage_joint_action: jnp.ndarray,
    zero_sensor_idx: jnp.ndarray,
) -> Tuple[EnvState, Params, RNGKey, DCRLTransition]:

    damaged_obs = env_state.obs.at[zero_sensor_idx].set(0.0)
    env_state = env_state.replace(obs=damaged_obs)

    actions = policy_network.apply(policy_params, env_state.obs)
    actions = actions.at[damage_joint_idx].set(damage_joint_action)

    state_desc = env_state.info["state_descriptor"]
    next_state = env.step(env_state, actions)

    transition = DCRLTransition(
        obs=env_state.obs,
        next_obs=next_state.obs,
        rewards=next_state.reward,
        dones=next_state.done,
        truncations=next_state.info["truncation"],
        actions=actions,
        state_desc=state_desc,
        next_state_desc=next_state.info["state_descriptor"],
        desc=jnp.zeros(
            env.descriptor_length,
        )
        * jnp.nan,
        desc_prime=jnp.zeros(
            env.descriptor_length,
        )
        * jnp.nan,
    )

    return next_state, policy_params, key, transition