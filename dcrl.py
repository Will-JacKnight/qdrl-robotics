import functools
import time
from typing import Any, Tuple

import jax
from jax.lax import fori_loop
import jax.numpy as jnp

import qdax.tasks.brax.v1 as environments
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids, compute_euclidean_centroids
from qdax.core.emitters.dcrl_me_emitter import DCRLMEConfig, DCRLMEEmitter
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.map_elites import MAPElites
from qdax.core.neuroevolution.buffers.buffer import DCRLTransition
from qdax.core.neuroevolution.networks.networks import MLP, MLPDC
from qdax.custom_types import EnvState, Params, RNGKey
from qdax.tasks.brax.v1 import descriptor_extractor
from qdax.tasks.brax.v1.wrappers.reward_wrappers import OffsetRewardWrapper, ClipRewardWrapper
from qdax.tasks.brax.v1.env_creators import scoring_function_brax_envs
from qdax.utils.metrics import default_qd_metrics, CSVLogger
from rollout import init_env_and_policy_network

def run_dcrl_map_elites(env_name,  #
             episode_length, #
             policy_hidden_layer_sizes,
             batch_size, #
             num_iterations, #
             grid_shape, #
             min_descriptor, #
             max_descriptor, #
             iso_sigma, #
             line_sigma, #
             ga_batch_size, #
             dcrl_batch_size, #
             ai_batch_size, #
             lengthscale, #
             critic_hidden_layer_size,
             num_critic_training_steps,
             num_pg_training_steps,
             replay_buffer_size,
             discount,
             reward_scaling, 
             critic_learning_rate,
             actor_learning_rate,
             policy_learning_rate,
             noise_clip,
             policy_noise,
             soft_tau_update,
             policy_delay,
             log_period, 
             key,
             dropout_rate,
             training_damage_rate,
):
    
    env, policy_network, actor_dc_network = init_env_and_policy_network(env_name, episode_length, policy_hidden_layer_sizes, dropout_rate)

    reset_fn = jax.jit(env.reset)
    
    # Init population of controllers
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num=batch_size)
    fake_batch_obs = jnp.zeros(shape=(batch_size, env.observation_size))
    init_params = jax.vmap(policy_network.init)(keys, fake_batch_obs)

    def play_step_fn(
        env_state: EnvState, policy_params: Params, key: RNGKey
    ) -> Tuple[EnvState, Params, RNGKey, DCRLTransition]:
        key, subkey = jax.random.split(key)
        actions = policy_network.apply(policy_params, env_state.obs, train=True, rngs={"dropout": subkey})
        state_desc = env_state.info["state_descriptor"]
        
        "random damage intensity"
        key, key_num_phy, key_num_sen, key_mask = jax.random.split(key, 4)
        num_phy_damage = jax.random.choice(key_num_phy, jnp.array([1, 2]), p=jnp.array([0.95, 0.05]))
        num_sen_damage = jax.random.choice(key_num_sen, jnp.array([1, 2, 3, 4]), p=jnp.array([0.5, 0.3, 0.15, 0.05]))

        damage_mask = (jax.random.uniform(key_mask) < training_damage_rate).astype(jnp.float32)

        def set_sensory_damage(i, carry):
            state, key = carry
            key, key_noise = jax.random.split(key)
            state_noise = jax.random.normal(key_noise, shape=state.obs.shape) * 0.05

            "reset states to sim sensory damage"
            key, key_state_idx, key_state_val = jax.random.split(key, 3)
            damage_sensor_idx = jax.random.randint(key_state_idx, shape=(), minval=0, maxval=27)
            damage_joint_action = jax.random.normal(key_state_val)

            new_val = state.obs[damage_sensor_idx] * (1 - damage_mask) + damage_joint_action * damage_mask
            damaged_obs = state.obs.at[damage_sensor_idx].set(new_val)
            state = state.replace(obs=damaged_obs + state_noise)
            return state, key
        
        def set_physical_damage(i, carry):
            actions, key = carry
            key, key_idx, key_val, key_noise = jax.random.split(key, 4)

            action_noise = jax.random.normal(key_noise, shape=actions.shape) * 0.05
            actions = jnp.clip(actions + action_noise, -1.0, 1.0)

            "action reset: physical damage sim"
            # uniform selection of damage joint
            damage_joint_idx = jax.random.randint(key_idx, shape=(), minval=0, maxval=actions.shape[0])
            damage_joint_action = jax.random.uniform(key_val, shape=(), minval=-1.0, maxval=1.0)

            "state/action variation"
            updated_actions = actions[damage_joint_idx] * (1 - damage_mask) + damage_joint_action * damage_mask
            updated_actions = jnp.clip(updated_actions, -1.0, 1.0)
            actions = actions.at[damage_joint_idx].set(updated_actions)
            return actions, key
        
        env_state, key = jax.lax.fori_loop(0, num_sen_damage, set_sensory_damage, (env_state, key))
        actions, key = jax.lax.fori_loop(0, num_phy_damage, set_physical_damage, (actions, key))

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
    
    # Prepare the scoring function
    descriptor_extraction_fn = descriptor_extractor[env_name]
    scoring_fn = functools.partial(
        scoring_function_brax_envs,
        episode_length=episode_length,
        play_reset_fn=reset_fn,
        play_step_fn=play_step_fn,
        descriptor_extractor=descriptor_extraction_fn,
    )

    # Get minimum reward value to make sure qd_score are positive
    reward_offset = environments.reward_offset[env_name]

    # Define a metrics function
    metrics_fn = functools.partial(
        default_qd_metrics,
        qd_offset=reward_offset * episode_length,
    )

    dcrl_emitter_config = DCRLMEConfig(
        ga_batch_size=ga_batch_size,
        dcrl_batch_size=dcrl_batch_size,
        ai_batch_size=ai_batch_size,
        lengthscale=lengthscale,
        critic_hidden_layer_size=critic_hidden_layer_size,
        num_critic_training_steps=num_critic_training_steps,
        num_pg_training_steps=num_pg_training_steps,
        batch_size=batch_size,
        replay_buffer_size=replay_buffer_size,
        discount=discount,
        reward_scaling=reward_scaling,
        critic_learning_rate=critic_learning_rate,
        actor_learning_rate=actor_learning_rate,
        policy_learning_rate=policy_learning_rate,
        noise_clip=noise_clip,
        policy_noise=policy_noise,
        soft_tau_update=soft_tau_update,
        policy_delay=policy_delay,
    )

    # Get the emitter
    variation_fn = functools.partial(
        isoline_variation, iso_sigma=iso_sigma, line_sigma=line_sigma
    )

    dcrl_emitter = DCRLMEEmitter(
        config=dcrl_emitter_config,
        policy_network=policy_network,
        actor_network=actor_dc_network,
        env=env,
        variation_fn=variation_fn,
    )

    # Instantiate MAP Elites
    map_elites = MAPElites(
        scoring_function=scoring_fn,
        emitter=dcrl_emitter,
        metrics_function=metrics_fn,
    )

    # Compute the centroids
    key, subkey = jax.random.split(key)
    # centroids = compute_cvt_centroids(
    #     num_descriptors=env.descriptor_length,
    #     num_init_cvt_samples=num_init_cvt_samples,
    #     num_centroids=num_centroids,
    #     minval=min_descriptor,
    #     maxval=max_descriptor,
    #     key=subkey,
    # )
    centroids = compute_euclidean_centroids(
        grid_shape=grid_shape,
        minval=min_descriptor,
        maxval=max_descriptor,
    )

    # compute initial repertoire
    key, subkey = jax.random.split(key)
    repertoire, emitter_state, init_metrics = map_elites.init(init_params, centroids, subkey)

    # Initialize metrics
    metrics = {key: jnp.array([]) for key in ["iteration", "qd_score", "coverage", "max_fitness", "time"]}

    # Set up init metrics
    init_metrics = jax.tree.map(lambda x: jnp.array([x]) if x.shape == () else x, init_metrics)
    init_metrics["iteration"] = jnp.array([0], dtype=jnp.int32)
    init_metrics["time"] = jnp.array([0.0])  # No time recorded for initialization

    # Convert init_metrics to match the metrics dictionary structure
    metrics = jax.tree.map(lambda metric, init_metric: jnp.concatenate([metric, init_metric], axis=0), metrics, init_metrics)

    # Initialise CSV logger
    # csv_logger = CSVLogger("repertoire_metrics.csv", header=list(metrics.keys()))

    # Main loop
    map_elites_scan_update = map_elites.scan_update
    num_loops = num_iterations // log_period
    for i in range(num_loops):
        start_time = time.time()
        (
            repertoire,
            emitter_state,
            key,
        ), current_metrics = jax.lax.scan(
            map_elites_scan_update,
            (repertoire, emitter_state, key),
            (),
            length=log_period,
        )
        timelapse = time.time() - start_time

        # Metrics
        current_metrics["iteration"] = jnp.arange(1+log_period*i, 1+log_period*(i+1), dtype=jnp.int32)
        current_metrics["time"] = jnp.repeat(timelapse, log_period)
        metrics = jax.tree.map(lambda metric, current_metric: jnp.concatenate([metric, current_metric], axis=0), metrics, current_metrics)

        # Log
        # csv_logger.log(jax.tree.map(lambda x:x[-1], metrics))

    return repertoire, metrics