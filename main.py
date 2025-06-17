import jax
import jax.numpy as jnp
import gpjax
import matplotlib.pyplot as plt
from utils.plot_results import plot_map_elites_results


from map_elites import run_map_elites
from adaptation import run_online_adaptation
from rollout import run_single_rollout


def main():
    fitness_drop_thres = 0.3

    env_name = 'ant_uni'            # reward = forward reward (proportional to forward velocity) + healthy reward - control cost - contact cost 
    episode_length = 500            # maximal rollout length
    seed = 42
    batch_size = 1024               # training batch for parallelisation
    num_iterations = 250
    grid_shape = (10, 10, 10, 10)
    min_descriptor = 0.0
    max_descriptor = 1.0
    policy_hidden_layer_sizes = (32, 32)
    iso_sigma = 0.005
    line_sigma = 0.05
    # num_init_cvt_samples = 50000
    # num_centroids = 1024
    log_period = 10

    # damage settings
    damage_joint_idx = 1
    damage_joint_action = 0 # value between [-1,1]


    key = jax.random.key(seed)
    key, subkey = jax.random.split(key)

    env, repertoire, metrics, policy_network = run_map_elites(env_name, episode_length, batch_size, num_iterations, grid_shape,
                   min_descriptor, max_descriptor, policy_hidden_layer_sizes,
                   iso_sigma, line_sigma, log_period, subkey)
    
    # plot map-elites results
    # env_steps = metrics["iteration"]
    fig, axes = plot_map_elites_results(num_iterations=jnp.arange(1, num_iterations+1), metrics=metrics, repertoire=repertoire, 
                                        min_bd=min_descriptor, max_bd=max_descriptor, grid_shape=grid_shape)
    plt.show()


    best_fitness = jnp.max(repertoire.fitnesses)
    best_idx = jnp.argmax(repertoire.fitnesses)
    params = jax.tree.map(lambda x: x[best_idx], repertoire.genotypes)

    key, subkey = jax.random.split(key)
    rollout = run_single_rollout(env, policy_network, params, subkey,
                                 damage_joint_idx=damage_joint_idx, damage_joint_action=damage_joint_action)
    
    key, subkey = jax.random.split(key)
    tested_indices, real_fitness, tested_goals = run_online_adaptation(repertoire, env, policy_network, subkey, 
                                                                        damage_joint_idx=damage_joint_idx, damage_joint_action=damage_joint_action)  #####
    print("********adaptation completes********")

    # real_fitness = rollout['rewards'].sum()
    # relative_fitness_diff = (best_fitness - real_fitness) / best_fitness

    # if relative_fitness_diff > fitness_drop_thres:
    #     print("********ite activated********")
    #     key, subkey = jax.random.split(key)
    #     tested_indices, real_fitness, tested_goals = run_online_adaptation(repertoire, env, policy_network, subkey, 
    #                           damage_joint_idx=damage_joint_idx, damage_joint_action=damage_joint_action)  #####
    #     print("********adaptation completes********")


if __name__ == "__main__":
    main()


    