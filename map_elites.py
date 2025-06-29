import functools
import time

import jax
import jax.numpy as jnp
import qdax.tasks.brax.v1 as environments
from qdax.core.containers.mapelites_repertoire import compute_euclidean_centroids
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.map_elites import MAPElites
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.tasks.brax.v1.env_creators import scoring_function_brax_envs as scoring_function
from qdax.utils.metrics import default_qd_metrics
from tqdm import trange

def init_env_and_policy_network(env_name, episode_length, policy_hidden_layer_sizes):
    """
    designed for adaptation tests only
    """
    # Init environment
    env = environments.create(env_name, episode_length=episode_length)

    # Init policy network
    policy_layer_sizes = policy_hidden_layer_sizes + (env.action_size,)
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )
    return env, policy_network


def run_map_elites(env_name, episode_length, policy_hidden_layer_sizes, batch_size, num_iterations, grid_shape,
                   min_descriptor, max_descriptor, iso_sigma, line_sigma, log_period, key):
    # Init environment
    env = environments.create(env_name, episode_length=episode_length)

    # Init policy network
    policy_layer_sizes = policy_hidden_layer_sizes + (env.action_size,)
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )
    reset_fn = jax.jit(env.reset)

    # Init population of controllers
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num=batch_size)
    fake_batch = jnp.zeros(shape=(batch_size, env.observation_size))
    init_variables = jax.vmap(policy_network.init)(keys, fake_batch)

    def play_step_fn(env_state, policy_params, key,):

        actions = policy_network.apply(policy_params, env_state.obs)

        state_desc = env_state.info["state_descriptor"]
        next_state = env.step(env_state, actions)

        transition = QDTransition(
            obs=env_state.obs,
            next_obs=next_state.obs,
            rewards=next_state.reward,
            dones=next_state.done,
            actions=actions,
            truncations=next_state.info["truncation"],
            state_desc=state_desc,
            next_state_desc=next_state.info["state_descriptor"],
        )

        return next_state, policy_params, key, transition
    # Define the scoring function
    descriptor_extraction_fn = environments.descriptor_extractor[env_name]
    scoring_fn = functools.partial(
        scoring_function,
        episode_length=episode_length,
        play_reset_fn=reset_fn,
        play_step_fn=play_step_fn,
        descriptor_extractor=descriptor_extraction_fn,
    )

    reward_offset = environments.reward_offset[env_name]

    # Define metrics function
    metrics_fn = functools.partial(
        default_qd_metrics,
        qd_offset=reward_offset * episode_length,
    )

    # Define emitter
    variation_fn = functools.partial(
        isoline_variation,
        iso_sigma=iso_sigma,         # isotropic Gaussian noise: mutation-like (explore the vicinity of existing elites)
        line_sigma=line_sigma,         # directional Gaussian noise: line-based, crossover-like (explore the vector connecting 2 elite solutions)
    )
    mixing_emitter = MixingEmitter(
        mutation_fn=None,
        variation_fn=variation_fn,
        variation_percentage=1.0,
        batch_size=batch_size,
    )

    # Instantiate MAP-Elites
    map_elites = MAPElites(
        scoring_function=scoring_fn,
        emitter=mixing_emitter,
        metrics_function=metrics_fn,
    )

    # Compute the centroids
    centroids = compute_euclidean_centroids(
        grid_shape=grid_shape,
        minval=min_descriptor,
        maxval=max_descriptor,
    )

    # Initializes repertoire and emitter state
    key, subkey = jax.random.split(key)
    repertoire, emitter_state, init_metrics = map_elites.init(init_variables, centroids, subkey)

    # Initialize metrics
    metrics = {metric_key: jnp.array([]) for metric_key in ["iteration", "qd_score", "coverage", "max_fitness", "time"]}

    # Set up init metrics
    init_metrics = jax.tree.map(lambda x: jnp.array([x]) if x.shape == () else x, init_metrics)
    init_metrics["iteration"] = jnp.array([0], dtype=jnp.int32)
    init_metrics["time"] = jnp.array([0.0])  # No time recorded for initialization

    # Convert init_metrics to match the metrics dictionary structure
    metrics = jax.tree.map(lambda metric, init_metric: jnp.concatenate([metric, init_metric], axis=0), metrics, init_metrics)


    map_elites_scan_update = map_elites.scan_update
    num_loops = num_iterations // log_period
    # Run MAP-Elites loop
    for i in trange(num_loops, desc="MAP Creation"):
        start_time = time.time()
        key, subkey = jax.random.split(key)
        (repertoire, emitter_state, key), current_metrics = jax.lax.scan(
            map_elites_scan_update,
            (repertoire, emitter_state, subkey),
            (), 
            length=log_period,
        )
        timelapse = time.time() - start_time

        current_metrics["iteration"] = jnp.arange(1+log_period*i, 1+log_period*(i+1), dtype=jnp.int32)
        current_metrics["time"] = jnp.repeat(timelapse, log_period)
        
        # Convert scalar metrics to 1D arrays for concatenation
        # current_metrics = jax.tree.map(lambda x: jnp.array([x]) if x.shape == () else x, current_metrics)
        metrics = jax.tree.map(lambda metric, current_metric: jnp.concatenate([metric, current_metric], axis=0), metrics, current_metrics)

    return repertoire, metrics, env, policy_network
