import argparse

import jax
import jax.numpy as jnp

from adaptation import run_online_adaptation
from map_elites import run_map_elites
from dcrl import run_dcrl_map_elites
from rollout import run_single_rollout
# from utils.plot_results import plot_map_elites_results
from utils.util import load_pkls, save_pkls
from utils.new_plot import plot_map_elites_results


def main(
         episode_length, 
         seed, 
         batch_size, 
         num_iterations, 
         grid_shape,
         policy_hidden_layer_sizes,
         damage_joint_idx, 
         damage_joint_action,
         ga_batch_size,
         dcrl_batch_size,
         ai_batch_size,
         lengthscale,
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
         output_path="./outputs",
         env_name='ant_uni',
         min_descriptor=0.0, 
         max_descriptor=1.0, 
         iso_sigma=0.005, 
         line_sigma=0.05, 
         log_period=10,
         ):
    

    key = jax.random.key(seed)
    key, subkey = jax.random.split(key)

    # map creation
    # repertoire, metrics, env, policy_network = run_map_elites(env_name, episode_length, policy_hidden_layer_sizes, batch_size, num_iterations, 
                                            # grid_shape, min_descriptor, max_descriptor, iso_sigma, line_sigma, log_period, subkey)
    
    repertoire, metrics, env, policy_network = run_dcrl_map_elites(env_name, episode_length, policy_hidden_layer_sizes, batch_size, num_iterations, 
                                            grid_shape, min_descriptor, max_descriptor, iso_sigma, line_sigma, ga_batch_size, 
                                            dcrl_batch_size, ai_batch_size, lengthscale, critic_hidden_layer_size, num_critic_training_steps,
                                            num_pg_training_steps, replay_buffer_size, discount, reward_scaling, critic_learning_rate,
                                            actor_learning_rate, policy_learning_rate, noise_clip, policy_noise, soft_tau_update,
                                            policy_delay, log_period, subkey)


    # save_pkls(output_path,repertoire=repertoire, metrics=metrics)
    # repertoire, metrics = load_pkls(output_path)
    breakpoint()
    # plot map-elites results
    env_steps = metrics["iteration"] * episode_length * batch_size
    plot_map_elites_results(env_steps=env_steps, metrics=metrics, repertoire=repertoire, 
                            min_bd=min_descriptor, max_bd=max_descriptor, grid_shape=grid_shape, output_dir=output_path)
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
    # breakpoint()

    key, subkey = jax.random.split(key)
    tested_indices, real_fitness, tested_goals = run_online_adaptation(repertoire, env, policy_network, subkey, 
                                                                       damage_joint_idx, damage_joint_action,)  #####
    print("********adaptation completes********")



def main_arg():
    parser = argparse.ArgumentParser(description="ITE Adaptation")
    # parser.add_argument("--output_path", type=str, default="./outputs", help="relative output path to project root (default)")
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
    # with open(pathlib.Path(args.output_path) / "config.json") as f:             ############fix
    #     json.dump(vars(args), f, indent=2)
    
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

    seed = 42

    output_path = "./outputs"
    log_period = 10

    env_name = 'ant_uni'            # reward = forward reward (proportional to forward velocity) + healthy reward - control cost - contact cost 
    episode_length = 1000            # maximal rollout length: 1000
    min_descriptor = 0.0
    max_descriptor = 1.0

    batch_size = 1024               # training batch for parallelisation: 1024
    num_iterations = 10                # 250, 500, 1000
    
    # Archive
    policy_hidden_layer_sizes = (32, 32)
    grid_shape = (5, 5, 5, 5)       # (10, 10, 10, 10)
    # num_init_cvt_samples = 50000
    # num_centroids = 1024

    # GA emitter
    iso_sigma = 0.005
    line_sigma = 0.05

    # DCRL-ME
    ga_batch_size = 128
    dcrl_batch_size = 64
    ai_batch_size = 64
    lengthscale = 0.1

    # DCRL emitter
    critic_hidden_layer_size = (256, 256)
    num_critic_training_steps = 3000
    num_pg_training_steps = 150
    replay_buffer_size = 1_024_000
    discount = 0.99
    reward_scaling = 1.0
    critic_learning_rate = 3e-4
    actor_learning_rate = 3e-4
    policy_learning_rate = 5e-3
    noise_clip = 0.5
    policy_noise = 0.2
    soft_tau_update = 0.005
    policy_delay = 2

    # damage settings: to achieve better results, compensatory behavior should be discovered in map
    damage_joint_idx = [0, 1]    # value between [0,7]
    damage_joint_action = [0, 0.9] # value between [-1,1]

    main(   
         episode_length, 
         seed, 
         batch_size, 
         num_iterations, 
         grid_shape,
         policy_hidden_layer_sizes,
         damage_joint_idx, 
         damage_joint_action,
         ga_batch_size,
         dcrl_batch_size,
         ai_batch_size,
         lengthscale,
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
         output_path,
         env_name,
         min_descriptor, 
         max_descriptor, 
         iso_sigma, 
         line_sigma, 
         log_period,
         )


if __name__ == "__main__":
    main_local()
    # main_arg()


    