import jax
import functools
import time
import jax.numpy as jnp
import matplotlib.pyplot as plt

from qdax.core.map_elites import MAPElites
from qdax.core.containers.mapelites_repertoire import compute_euclidean_centroids
from qdax import environments
from qdax.tasks.brax_envs import scoring_function_brax_envs as scoring_function
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.utils.metrics import default_qd_metrics
from qdax.utils.plotting import plot_map_elites_results

from rollout import run_single_rollout
from utils.util import save_pkls


def run_map_elites(env_name, episode_length, batch_size, num_iterations, grid_shape,
                   min_descriptor, max_descriptor, policy_hidden_layer_sizes,
                   iso_sigma, line_sigma, log_period, key):
    # Init environment
    env = environments.create(env_name, episode_length=episode_length)


    key, subkey = jax.random.split(key)

    # Init policy network
    policy_layer_sizes = policy_hidden_layer_sizes + (env.action_size,)
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )

    # Init population of controllers
    keys = jax.random.split(subkey, num=batch_size)
    fake_batch = jnp.zeros(shape=(batch_size, env.observation_size))
    init_variables = jax.vmap(policy_network.init)(keys, fake_batch)

    # Init environment state
    key, subkey = jax.random.split(key)
    keys = jnp.repeat(jnp.expand_dims(subkey, axis=0), repeats=batch_size, axis=0)
    reset_fn = jax.jit(jax.vmap(env.reset))
    init_states = reset_fn(keys)


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
    descriptor_extraction_fn = environments.behavior_descriptor_extractor[env_name]
    scoring_fn = functools.partial(
        scoring_function,
        init_states=init_states,
        episode_length=episode_length,
        play_step_fn=play_step_fn,
        behavior_descriptor_extractor=descriptor_extraction_fn,
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
        # minval=min_param,
        # maxval=max_param,
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
    repertoire, emitter_state, key = map_elites.init(init_variables, centroids, subkey)


    # Initialize metrics
    metrics = {key: jnp.array([]) for key in ["iteration", "qd_score", "coverage", "max_fitness", "time"]}

    # # Set up init metrics
    # init_metrics = jax.tree.map(lambda x: jnp.array([x]) if x.shape == () else x, init_metrics)
    # metrics["iteration"] = jnp.array([0], dtype=jnp.int32)
    # metrics["time"] = jnp.array([0.0])  # No time recorded for initialization

    # metrics = jax.tree.map(lambda metric, init_metric: jnp.concatenate([metric, init_metric], axis=0), metrics, init_metrics)

    # # Jit the update function for faster iterations
    # update_fn = jax.jit(map_elites.update)

    map_elites_scan_update = map_elites.scan_update

    # Run MAP-Elites loop
    for i in range(num_iterations // log_period):
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

    return env, repertoire, metrics, policy_network


    
if __name__ == "__main__":
    env_name = 'ant_uni'
    episode_length = 100            # maximal rollout length
    seed = 42
    batch_size = 100                # training batch for parallelisation
    num_iterations = 50
    grid_shape = (10, 10, 10, 10)
    # min_param = 0.0
    # max_param = 1.0
    min_descriptor = 0.0
    max_descriptor = 1.0
    policy_hidden_layer_sizes = (32, 32)
    iso_sigma = 0.005
    line_sigma = 0.05
    # num_init_cvt_samples = 50000
    # num_centroids = 1024
    log_period = 10

    key = jax.random.key(seed)

    env, repertoire, metrics, policy_network = run_map_elites(env_name, episode_length, batch_size, num_iterations, grid_shape,
                   min_descriptor, max_descriptor, policy_hidden_layer_sizes,
                   iso_sigma, line_sigma, log_period, key)
    
    ## plot coverage results
    # env_steps = metrics["iteration"]
    # fig, axes = plot_map_elites_results(env_steps=env_steps, metrics=metrics, repertoire=repertoire, min_descriptor=min_descriptor, max_descriptor=max_descriptor)
    # plt.show()

    best_idx = jnp.argmax(repertoire.fitnesses)
    best_fitness = jnp.max(repertoire.fitnesses)
    best_descriptor = repertoire.descriptors[best_idx]
    print(
        f"Best fitness in the repertoire: {best_fitness:.2f}\n",
        f"Descriptor of the best individual in the repertoire: {best_descriptor}\n",
        f"Index in the repertoire of this individual: {best_idx}\n"
    )

    # select the parameters of the best individual
    my_params = jax.tree.map(
        lambda x: x[best_idx],
        repertoire.genotypes
    )

    # single rollout with best niche
    rollout = run_single_rollout(env, policy_network, my_params, key)
    print(len(rollout['rewards']))
    print(rollout['rewards'])

    save_pkls(repertoire=repertoire, metrics=metrics)
