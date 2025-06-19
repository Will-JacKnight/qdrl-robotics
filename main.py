import jax
import jax.numpy as jnp
import gpjax
import matplotlib.pyplot as plt

from qdax import environments
from qdax.core.neuroevolution.networks.networks import MLP
from matplotlib.ticker import ScalarFormatter

from map_elites import run_map_elites, init_env_and_policy
from adaptation import run_online_adaptation
from rollout import run_single_rollout

from utils.util import save_pkls, load_pkls
from utils.plot_results import plot_map_elites_results




def main(env_name, episode_length, seed, batch_size, num_iterations, grid_shape,
         min_descriptor, max_descriptor, policy_hidden_layer_sizes,
         iso_sigma, line_sigma, log_period, damage_joint_idx, damage_joint_action):
    
    # init brax environment
    env, policy_network = init_env_and_policy(env_name, episode_length, policy_hidden_layer_sizes)

    key = jax.random.key(seed)
    key, subkey = jax.random.split(key)

    # repertoire, metrics = run_map_elites(env, policy_network, batch_size, num_iterations, grid_shape,
    #                min_descriptor, max_descriptor, iso_sigma, line_sigma, log_period, subkey)
    
    # save_pkls(repertoire=repertoire, metrics=metrics)
    repertoire, metrics = load_pkls()

    # plot map-elites results
    # env_steps = metrics["iteration"]
    fig, axes = plot_map_elites_results(num_iterations=jnp.arange(1, num_iterations+1), metrics=metrics, repertoire=repertoire, 
                                        min_bd=min_descriptor, max_bd=max_descriptor, grid_shape=grid_shape)
    
    axes.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    axes.yaxis.get_major_formatter().set_scientific(False)
    axes.yaxis.get_major_formatter().set_useOffset(False)

    axes.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    axes.xaxis.get_major_formatter().set_scientific(False)
    axes.xaxis.get_major_formatter().set_useOffset(False)
    plt.show(block=False)

    breakpoint()
    
    best_fitness = jnp.max(repertoire.fitnesses)
    best_idx = jnp.argmax(repertoire.fitnesses)
    best_descriptor = repertoire.descriptors[best_idx]
    params = jax.tree.map(lambda x: x[best_idx], repertoire.genotypes)

    print(
        "Intact training without damage: \n",
        f"Best fitness: {best_fitness:.2f}\n",
        f"Best descriptor: {best_descriptor}\n",
        f"Index of best fitness niche: {best_idx}\n"
    )

    key, subkey = jax.random.split(key)
    rollout = run_single_rollout(env, policy_network, params, subkey, 
                                 None, None,
                                 "./outputs/pre_adaptation_without_damage.html")
    # breakpoint()
    key, subkey = jax.random.split(key)
    rollout = run_single_rollout(env, policy_network, params, subkey, 
                                 damage_joint_idx, damage_joint_action, 
                                 "./outputs/pre_adaptation_with_damage.html")
    
    # breakpoint()
    key, subkey = jax.random.split(key)
    tested_indices, real_fitness, tested_goals = run_online_adaptation(repertoire, env, policy_network, subkey, 
                                                                       damage_joint_idx, damage_joint_action)  #####
    print("********adaptation completes********")

    # real_fitness = rollout['rewards'].sum()
    # relative_fitness_diff = (best_fitness - real_fitness) / best_fitness

    # if relative_fitness_diff > fitness_drop_thres:
    #     print("********ite activated********")
    #     key, subkey = jax.random.split(key)
    #     tested_indices, real_fitness, tested_goals = run_online_adaptation(repertoire, env, policy_network, subkey, 
    #                           damage_joint_idx=damage_joint_idx, damage_joint_action=damage_joint_action)  #####
    #     print("********adaptation completes********")


def main_arg():
    pass


if __name__ == "__main__":

    env_name = 'ant_uni'            # reward = forward reward (proportional to forward velocity) + healthy reward - control cost - contact cost 
    episode_length = 1000            # maximal rollout length
    seed = 42
    batch_size = 1024               # training batch for parallelisation
    num_iterations = 250                # 250, 500, 1000
    grid_shape = (10, 10, 10, 10)
    min_descriptor = 0.0
    max_descriptor = 1.0
    policy_hidden_layer_sizes = (32, 32)
    iso_sigma = 0.005
    line_sigma = 0.05
    # num_init_cvt_samples = 50000
    # num_centroids = 1024
    log_period = 10

    # damage settings: to achieve better results, compensatory behavior should be discovered in map
    damage_joint_idx = [0, 1]    # value between [0,7]
    damage_joint_action = [0, 0.9] # value between [-1,1]
    # fitness_drop_thres = 0.3

    main(env_name, episode_length, seed, batch_size, num_iterations, grid_shape,
         min_descriptor, max_descriptor, policy_hidden_layer_sizes,
         iso_sigma, line_sigma, log_period, damage_joint_idx, damage_joint_action)


    