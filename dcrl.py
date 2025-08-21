import functools
import time
from typing import Any, Tuple

import jax
from jax.lax import fori_loop
import jax.numpy as jnp

import qdax.tasks.brax.v1 as environments
from qdax.core.containers.mapelites_repertoire import compute_euclidean_centroids
from qdax.core.emitters.dcrl_me_emitter import DCRLMEConfig, DCRLMEEmitter
from qdax.core.emitters.mutation_operators import isoline_variation
# from qdax.core.map_elites import MAPElites
from qdax.core.neuroevolution.buffers.buffer import DCRLTransition
from qdax.core.neuroevolution.networks.networks import MLP, MLPDC
from qdax.custom_types import EnvState, Params, RNGKey
from qdax.tasks.brax.v1 import descriptor_extractor
from qdax.tasks.brax.v1.wrappers.reward_wrappers import OffsetRewardWrapper, ClipRewardWrapper
from qdax.tasks.brax.v1.env_creators import scoring_function_brax_envs
from qdax.utils.metrics import default_qd_metrics, CSVLogger

from rollout import init_env_and_policy_network
from setup_containers import setup_container
from core.mapelites_sampling import ReevalMAPElites

def run_dcrl_map_elites(
    env_name: str,
    container_name: str,
    episode_length: int,
    policy_hidden_layer_sizes: Tuple[int, ...],
    batch_size: int,
    num_iterations: int, #
    grid_shape: Tuple[int, ...],
    min_descriptor, 
    max_descriptor, 
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
    num_evals,
    depth: int,
    max_number_evals: int,
    fitness_extractor_method: str, 
    fitness_reproducibility_extractor_method: str, 
    descriptor_extractor_method: str, 
    descriptor_reproducibility_extractor_method: str,
    as_repertoire_num_samples: int,
    extract_type: str,
    sampling_size: int,
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
    
    def multi_eval_play_step_fn(
        env_state: EnvState, policy_params: Params, key: RNGKey, num_evals: int
    ) -> Tuple[EnvState, Params, RNGKey, DCRLTransition]:
        "eval and average the transition"
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, num_evals)
        batched_play_step_fn = jax.vmap(lambda k: play_step_fn(env_state, policy_params, k))
        next_states, batched_policy_params, _, transitions = batched_play_step_fn(keys)

        # get the first out of the batched results
        next_state = jax.tree.map(lambda x:x[0], next_states)
        policy_params = jax.tree.map(lambda x:x[0], batched_policy_params)
        # average the rewards
        final_transition = jax.tree.map(lambda x:x[0], transitions)
        final_transition = final_transition.replace(rewards=jnp.mean(transitions.rewards))
        return next_state, policy_params, key, final_transition
    
    avg_eval_fn = functools.partial(multi_eval_play_step_fn, num_evals=num_evals)
    
    # Prepare the scoring function
    descriptor_extraction_fn = descriptor_extractor[env_name]
    scoring_fn = functools.partial(
        scoring_function_brax_envs,
        episode_length=episode_length,
        play_reset_fn=reset_fn,
        play_step_fn=play_step_fn,       # avg_eval_fn
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
    # map_elites = ReevalMAPElites(
    #     num_evals=num_evals,
    #     scoring_function=scoring_fn,
    #     emitter=dcrl_emitter,
    #     metrics_function=metrics_fn,
    # )
    key, subkey = jax.random.split(key)
    map_elites, key = setup_container(
        container_name=container_name,
        emitter=dcrl_emitter,
        num_samples=num_evals,
        depth=depth,
        scoring_function=scoring_fn,
        metrics_function=metrics_fn,
        sampling_size=sampling_size,
        max_number_evals=max_number_evals,
        as_repertoire_num_samples=as_repertoire_num_samples,
        fitness_extractor=fitness_extractor_method, 
        fitness_reproducibility_extractor=fitness_reproducibility_extractor_method, 
        descriptor_extractor=descriptor_extractor_method, 
        descriptor_reproducibility_extractor=descriptor_reproducibility_extractor_method,
        extract_type=extract_type,
        key = subkey,
    )

    # Compute the centroids
    key, subkey = jax.random.split(key)
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