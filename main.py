import argparse
import jax
import jax.numpy as jnp
import gpjax
import matplotlib.pyplot as plt
import pathlib
import json

from qdax import environments
from qdax.core.neuroevolution.networks.networks import MLP

from map_creation import run_map_elites, init_env_and_policy
from adaptation import run_online_adaptation
from rollout import run_single_rollout

from utils.util import save_pkls, load_pkls
from utils.plot_results import plot_map_elites_results



def main(
         episode_length, 
         seed, 
         batch_size, 
         num_iterations, 
         grid_shape,
         policy_hidden_layer_sizes,
         damage_joint_idx, 
         damage_joint_action,
         env_name='ant_uni',
         min_descriptor=0.0, 
         max_descriptor=1.0, 
         iso_sigma=0.005, 
         line_sigma=0.05, 
         log_period=10
         ):
    
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

    # key, subkey = jax.random.split(key)
    # rollout = run_single_rollout(env, policy_network, params, subkey, 
    #                              None, None,
    #                              "./outputs/pre_adaptation_without_damage.html")
    # breakpoint()
    key, subkey = jax.random.split(key)
    rollout = run_single_rollout(env, policy_network, params, subkey, 
                                 damage_joint_idx, damage_joint_action, 
                                 "./outputs/pre_adaptation_with_damage.html")
    
    breakpoint()
    key, subkey = jax.random.split(key)
    tested_indices, real_fitness, tested_goals = run_online_adaptation(repertoire, env, policy_network, subkey, 
                                                                       damage_joint_idx, damage_joint_action,)  #####
    print("********adaptation completes********")



def main_arg():
    parser = argparse.ArgumentParser(description="ITE Adaptation")
    parser.add_argument("--output_path", type=str, default="./outputs", help="relative output path to project root (default)")
    parser.add_argument("--episode_length", type=int, default=1000, help="Maximum rollout length (default: 1000)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for seed generation")
    parser.add_argument("--batch_size", type=int, default=1024, help="Training batch size (default: 1024)")
    parser.add_argument("--num_iterations", type=int, default=250, help="Number of training iterations (default: 250)")
    parser.add_argument("--grid_shape", type=int, nargs='+', help="Shape of the MAP grid, use format: --grid_shape 10 10 10 10")
    parser.add_argument("--policy_hidden_layer_sizes", type=int, nargs='+', help="Hidden layer size of controller policy")
    parser.add_argument("--damage_joint_idx", type=int, nargs='+', help="Index of the damaged joint")
    parser.add_argument("--damage_joint_action", type=int, nargs='+', help="Action value of the damaged joint")
    args = parser.parse_args()

    if len(args.damage_joint_idx) != len(args.damage_joint_action):
        raise ValueError("Number of damage joint actions need to match the number of damage joint indices.")

    # save the args to config.json
    with open(pathlib.Path(args.output_path) / "config.json") as f:
        json.dump(vars(args), f, indent=2)
    
    main(
         args.episode_length, 
         args.seed, 
         args.batch_size, 
         args.num_iterations, 
         args.grid_shape,
         args.policy_hidden_layer_sizes,
         args.damage_joint_idx, 
         args.damage_joint_action
        )

def main_local():
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
    damage_joint_idx = [5, 7]    # value between [0,7]
    damage_joint_action = [0, 0] # value between [-1,1]

    main(   
         episode_length, 
         seed, 
         batch_size, 
         num_iterations, 
         grid_shape,
         policy_hidden_layer_sizes,
         damage_joint_idx, 
         damage_joint_action,
         env_name,
         min_descriptor, 
         max_descriptor, 
         iso_sigma, 
         line_sigma, 
         log_period
         )


if __name__ == "__main__":
    main_local()
    # main_arg()


    