import jax
import jax.numpy as jnp
import gpjax

breakpoint()
from map_elites import run_map_elites
from adaptation import run_online_adaptation
from rollout import run_single_rollout


def main():
    fitness_drop_thres = 0.3

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

    env, repertoire, _, policy_network = run_map_elites(env_name, episode_length, batch_size, num_iterations, grid_shape,
                   min_descriptor, max_descriptor, policy_hidden_layer_sizes,
                   iso_sigma, line_sigma, log_period, key)
    
    best_fitness = jnp.max(repertoire.fitnesses)
    best_idx = jnp.argmax(repertoire.fitnesses)
    params = jax.tree.map(lambda x: x[best_idx], repertoire.genotypes)

    rollout = run_single_rollout(env, policy_network, params, key)

    real_fitness = rollout['rewards'].sum()
    relative_fitness_diff = (best_fitness - real_fitness) / best_fitness

    if relative_fitness_diff > fitness_drop_thres:
        run_online_adaptation(repertoire, env, policy_network, key)  #####


if __name__ == "__main__":
    main()


    