import functools
import time
from typing import Any, Tuple

import jax
import jax.numpy as jnp

import qdax.tasks.brax.v1 as environments
from qdax.core.containers.mapelites_repertoire import compute_euclidean_centroids
from qdax.core.emitters.dcrl_me_emitter import DCRLMEConfig, DCRLMEEmitter
from qdax.core.emitters.mutation_operators import isoline_variation
# from qdax.core.map_elites import MAPElites
from qdax.core.neuroevolution.buffers.buffer import DCRLTransition
from qdax.custom_types import EnvState, Params, RNGKey
from qdax.tasks.brax.v1 import descriptor_extractor
from qdax.tasks.brax.v1.env_creators import scoring_function_brax_envs
from qdax.utils.metrics import default_qd_metrics, CSVLogger
from core.containers.mapelites_repertoire import MapElitesRepertoire

from rollout import init_env_and_policy_network
from setup_containers import setup_container, EXTRACTOR_LIST
from utils.uncertainty_metrics import reevaluation_function


def run_dcrl_map_elites(args: Any, key: RNGKey):
    
    env, policy_network, actor_dc_network = init_env_and_policy_network(
        env_name=args.env_name, 
        episode_length=args.episode_length, 
        policy_hidden_layer_sizes=args.policy_hidden_layer_sizes, 
        dropout_rate=args.dropout_rate
    )

    reset_fn = jax.jit(env.reset)
    
    # Init population of controllers
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num=args.init_batch_size)
    fake_batch_obs = jnp.zeros(shape=(args.init_batch_size, env.observation_size))
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
        env_state: EnvState, policy_params: Params, key: RNGKey, num_samples: int
    ) -> Tuple[EnvState, Params, RNGKey, DCRLTransition]:
        "eval and average the transition"
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, num_samples)
        batched_play_step_fn = jax.vmap(lambda k: play_step_fn(env_state, policy_params, k))
        next_states, batched_policy_params, _, transitions = batched_play_step_fn(keys)

        # get the first out of the batched results
        next_state = jax.tree.map(lambda x:x[0], next_states)
        policy_params = jax.tree.map(lambda x:x[0], batched_policy_params)
        # average the rewards
        final_transition = jax.tree.map(lambda x:x[0], transitions)
        final_transition = final_transition.replace(rewards=jnp.mean(transitions.rewards))
        return next_state, policy_params, key, final_transition
    
    avg_eval_fn = functools.partial(multi_eval_play_step_fn, num_samples=args.num_samples)
    
    # Prepare the scoring function
    descriptor_extraction_fn = descriptor_extractor[args.env_name]
    scoring_fn = functools.partial(
        scoring_function_brax_envs,
        episode_length=args.episode_length,
        play_reset_fn=reset_fn,
        play_step_fn=play_step_fn,       # avg_eval_fn
        descriptor_extractor=descriptor_extraction_fn,
    )

    # Get minimum reward value to make sure qd_score are positive
    reward_offset = environments.reward_offset[args.env_name]

    # Define a metrics function
    metrics_fn = functools.partial(
        default_qd_metrics,
        qd_offset=reward_offset * args.episode_length,
    )

    dcrl_emitter_config = DCRLMEConfig(
        ga_batch_size=args.ga_batch_size,
        dcrl_batch_size=args.dcrl_batch_size,
        ai_batch_size=args.ai_batch_size,
        lengthscale=args.lengthscale,
        critic_hidden_layer_size=args.critic_hidden_layer_size,
        num_critic_training_steps=args.num_critic_training_steps,
        num_pg_training_steps=args.num_pg_training_steps,
        batch_size=args.batch_size,
        replay_buffer_size=args.replay_buffer_size,
        discount=args.discount,
        reward_scaling=args.reward_scaling,
        critic_learning_rate=args.critic_learning_rate,
        actor_learning_rate=args.actor_learning_rate,
        policy_learning_rate=args.policy_learning_rate,
        noise_clip=args.noise_clip,
        policy_noise=args.policy_noise,
        soft_tau_update=args.soft_tau_update,
        policy_delay=args.policy_delay,
    )

    # Get the emitter
    variation_fn = functools.partial(
        isoline_variation, iso_sigma=args.iso_sigma, line_sigma=args.line_sigma
    )

    dcrl_emitter = DCRLMEEmitter(
        config=dcrl_emitter_config,
        policy_network=policy_network,
        actor_network=actor_dc_network,
        env=env,
        variation_fn=variation_fn,
    )

    # Instantiate MAP Elites
    key, subkey = jax.random.split(key)
    map_elites, key = setup_container(
        container=args.container,
        emitter=dcrl_emitter,
        num_samples=args.num_samples,
        depth=args.depth,
        scoring_function=scoring_fn,
        metrics_function=metrics_fn,
        batch_size=args.batch_size,
        emit_batch_size=args.emit_batch_size,
        max_number_evals=args.max_number_evals,
        as_repertoire_num_samples=args.as_repertoire_num_samples,
        fitness_extractor=args.fitness_extractor, 
        fitness_reproducibility_extractor=args.fitness_reproducibility_extractor, 
        descriptor_extractor=args.descriptor_extractor, 
        descriptor_reproducibility_extractor=args.descriptor_reproducibility_extractor,
        extract_type=args.extract_type,
        key = subkey,
    )

    # Compute the centroids
    key, subkey = jax.random.split(key)
    centroids = compute_euclidean_centroids(
        grid_shape=args.grid_shape,
        minval=args.min_descriptor,
        maxval=args.max_descriptor,
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

    # Create an empty standard MapElitesRepertoire for metrics
    # metrics_repertoire = MapElitesRepertoire.init(
    #     genotypes=init_params,
    #     fitnesses=jnp.zeros(args.init_batch_size),
    #     descriptors=jnp.zeros((args.init_batch_size, centroids.shape[-1])),
    #     extra_scores={},
    #     centroids=centroids,
    # )
    
    # def corrected_scan_update(
    #     carry: Tuple[Any, Any, RNGKey],
    #     _: Any,
    # ) -> Tuple[Tuple[Any, Any, RNGKey], Any]:
    #     """
    #     Custom scan update that performs both regular update and correction at each step.
    #     """
    #     repertoire, emitter_state, key = carry
        
    #     # Perform regular MAP-Elites update
    #     key, subkey = jax.random.split(key)
    #     repertoire, emitter_state, regular_metrics = map_elites.update(
    #         repertoire, emitter_state, subkey
    #     )
        
    #     # Perform reevaluation and correction
    #     key, subkey = jax.random.split(key)
    #     corrected_repertoire, key = reevaluation_function(
    #         repertoire=repertoire,
    #         random_key=subkey,
    #         metric_repertoire=repertoire,
    #         scoring_fn=scoring_fn,
    #         num_reevals=args.num_reevals,
    #         scan_size=args.reeval_scan_size,
    #         fitness_extractor=EXTRACTOR_LIST[args.reeval_fitness_extractor],
    #         fitness_reproducibility_extractor=EXTRACTOR_LIST[args.reeval_fitness_reproducibility_extractor],
    #         descriptor_extractor=EXTRACTOR_LIST[args.reeval_descriptor_extractor],
    #         descriptor_reproducibility_extractor=EXTRACTOR_LIST[args.reeval_descriptor_reproducibility_extractor],
    #     )

    #     # Compute corrected metrics
    #     corrected_metrics = metrics_fn(corrected_repertoire)
    #     # corrected_metrics = jax.tree.map(lambda x: jnp.array([x]) if x.shape == () else x, corrected_metrics)
        
        
    #     return (repertoire, emitter_state, key), corrected_metrics


    # # Main loop
    # num_loops = args.num_iterations // args.log_period
    # for i in range(num_loops):
    #     start_time = time.time()
    #     (
    #         repertoire,
    #         emitter_state,
    #         key,
    #     ), current = jax.lax.scan(
    #         map_elites.scan_update
    #         # corrected_scan_update,      
    #         (repertoire, emitter_state, key),
    #         (),
    #         length=args.log_period,
    #     )
    #     timelapse = time.time() - start_time

    #     # Metrics
    #     corrected_metrics["iteration"] = jnp.arange(1+args.log_period*i, 1+args.log_period*(i+1), dtype=jnp.int32)
    #     corrected_metrics["time"] = jnp.repeat(timelapse, args.log_period)
    #     metrics = jax.tree.map(lambda metric, current_metric: jnp.concatenate([metric, current_metric], axis=0), metrics, corrected_metrics)

    #     # Log
    #     # csv_logger.log(jax.tree.map(lambda x:x[-1], metrics))

    
        # Main loop
    for i in range(args.num_iterations):
        start_time = time.time()
        (
            repertoire,
            emitter_state,
            current_metrics
        ) = map_elites.update(repertoire, emitter_state, key)

        if i % args.log_period != 0:
            continue
        
        timelapse = time.time() - start_time

        # Perform reevaluation and correction
        key, subkey = jax.random.split(key)
        corrected_repertoire, key = reevaluation_function(
            repertoire=repertoire,
            random_key=subkey,
            metric_repertoire=repertoire,
            scoring_fn=scoring_fn,
            num_reevals=args.num_reevals,
            scan_size=args.reeval_scan_size,
            fitness_extractor=EXTRACTOR_LIST[args.reeval_fitness_extractor],
            fitness_reproducibility_extractor=EXTRACTOR_LIST[args.reeval_fitness_reproducibility_extractor],
            descriptor_extractor=EXTRACTOR_LIST[args.reeval_descriptor_extractor],
            descriptor_reproducibility_extractor=EXTRACTOR_LIST[args.reeval_descriptor_reproducibility_extractor],
        )

        # Compute corrected metrics
        corrected_metrics = metrics_fn(corrected_repertoire)
        # Metrics
        corrected_metrics["iteration"] = jnp.arange(1+args.log_period*i, 1+args.log_period*(i+1), dtype=jnp.int32)
        corrected_metrics["time"] = jnp.repeat(timelapse, args.log_period)
        metrics = jax.tree.map(lambda metric, current_metric: jnp.concatenate([metric, current_metric], axis=0), metrics, corrected_metrics)


    return repertoire, metrics