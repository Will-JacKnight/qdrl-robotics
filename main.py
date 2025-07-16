from typing import Any, Dict, Iterable, List, Optional, Tuple, Literal
import argparse
import json
import sys
from datetime import datetime

import jax
import jax.numpy as jnp

from adaptation import run_online_adaptation
# from adaptation_tgp import run_online_adaptation
from map_elites import run_map_elites
from dcrl import run_dcrl_map_elites
from rollout import run_single_rollout, init_env_and_policy_network, render_rollout_to_html
from utils.util import load_pkls, save_pkls, save_args
from utils.new_plot import plot_map_elites_results


def main(
         mode: Literal["training", "adaptation"],
         algo_type,
         episode_length, 
         seed, 
         batch_size, 
         num_iterations: int, 
         grid_shape: Tuple[int, ...],
         policy_hidden_layer_sizes: Tuple[int, ...],
         damage_joint_idx: jnp.ndarray, 
         damage_joint_action: jnp.ndarray,
         zero_sensor_idx: jnp.ndarray, 
         ga_batch_size,
         dcrl_batch_size,
         ai_batch_size,
         lengthscale,
         critic_hidden_layer_size: Tuple[int, ...],
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
         output_path: str,
         exp_path: str,
         env_name: str,
         min_descriptor, 
         max_descriptor, 
         iso_sigma: int, 
         line_sigma: int, 
         log_period: int,
         max_iters: int, 
         performance_threshold
         ):
    

    key = jax.random.key(seed)
    key, subkey = jax.random.split(key)


    # map creation
    if mode == "training":
        match algo_type:
            case "mapelites":
                repertoire, metrics = run_map_elites(env_name, episode_length, policy_hidden_layer_sizes, batch_size, num_iterations, 
                                                grid_shape, min_descriptor, max_descriptor, iso_sigma, line_sigma, log_period, subkey)
            case "dcrl":
                repertoire, metrics = run_dcrl_map_elites(env_name, episode_length, policy_hidden_layer_sizes, batch_size, num_iterations, 
                                                grid_shape, min_descriptor, max_descriptor, iso_sigma, line_sigma, ga_batch_size, 
                                                dcrl_batch_size, ai_batch_size, lengthscale, critic_hidden_layer_size, num_critic_training_steps,
                                                num_pg_training_steps, replay_buffer_size, discount, reward_scaling, critic_learning_rate,
                                                actor_learning_rate, policy_learning_rate, noise_clip, policy_noise, soft_tau_update,
                                                policy_delay, log_period, subkey)
            case _:
                raise ValueError(f"Unknown algo_type: {algo_type}")

        save_pkls(output_path, repertoire=repertoire, metrics=metrics)

        "remove the following inside plot function"
        env_steps = metrics["iteration"] * episode_length * batch_size
        plot_map_elites_results(env_steps=env_steps, metrics=metrics, repertoire=repertoire, 
                                min_bd=min_descriptor, max_bd=max_descriptor, grid_shape=grid_shape, output_dir=output_path)


    repertoire, metrics = load_pkls(output_path)
    env, policy_network = init_env_and_policy_network(env_name, episode_length, policy_hidden_layer_sizes)
 
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

    if mode == "training":
        key, subkey = jax.random.split(key)
        rollout = run_single_rollout(env, policy_network, params, subkey)
        render_rollout_to_html(rollout['states'], env, output_path + "/pre_adaptation_without_damage.html")
        

    key, subkey = jax.random.split(key)
    rollout = run_single_rollout(env, policy_network, params, subkey, 
                                 damage_joint_idx, damage_joint_action, zero_sensor_idx)
    render_rollout_to_html(rollout['states'], env, exp_path + "/pre_adaptation_with_damage.html")

    key, subkey = jax.random.split(key)
    run_online_adaptation(repertoire, env, policy_network, subkey, exp_path, min_descriptor, max_descriptor, grid_shape, 
                          damage_joint_idx, damage_joint_action, zero_sensor_idx,
                          episode_length, max_iters, performance_threshold)



def get_args():
    parser = argparse.ArgumentParser(description="ITE Adaptation")
    parser.add_argument("--config", type=str, default="./config.json",help='Path to config.json (default parameter file)')
    args, remaining_args = parser.parse_known_args()

    with open(args.config, 'r') as f:
        config_args = json.load(f)

    parser.set_defaults(**config_args)
    parser.add_argument("--mode", type=str, help="run mode: training || adaptation (default)")
    parser.add_argument("--output_path", type=str, help="relative path to the MAP")
    parser.add_argument("--exp_path", type=str, help="relative path to specific damage runs")
    parser.add_argument("--algo_type", type=str, help="choose from: mapelites || dcrl")
    parser.add_argument("--episode_length", type=int, help="Maximum rollout length")
    parser.add_argument("--batch_size", type=int, help="Training batch size")
    parser.add_argument("--num_iterations", type=int, help="Number of training iterations")
    parser.add_argument("--grid_shape", type=int, nargs='+', help="Shape of the MAP grid, use format: --grid_shape 10 10 10 10")
    parser.add_argument("--policy_hidden_layer_sizes", type=int, nargs='+', help="Hidden layer size of controller policy")
    parser.add_argument("--damage_joint_idx", type=int, nargs='+', help="Index of the damaged joint")
    parser.add_argument("--damage_joint_action", type=float, nargs='+', help="Action value of the damaged joint")
    parser.add_argument("--zero_sensor_idx", type=int, nargs='+', help="Index of the zero sensor")
    parser.add_argument("--damage_type", type=str, help="Damage type: physical | sensory")
    args = parser.parse_args()

    args.grid_shape = tuple(args.grid_shape)
    args.policy_hidden_layer_sizes = tuple(args.policy_hidden_layer_sizes)
    args.critic_hidden_layer_size = tuple(args.critic_hidden_layer_size)

    if len(args.damage_joint_idx) != len(args.damage_joint_action):
        raise ValueError("Number of damage joint actions need to match the number of damage joint indices.")

    print(f"algo type: {args.algo_type}")
    run_mode = args.mode
    print(f"Running on: {run_mode} mode")
    if run_mode == "training":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_path = args.output_path + f"/{args.algo_type}_{timestamp}"
        print(f"starting time: {timestamp}")

        if "--algo_type" not in sys.argv:
            raise ValueError("You must specify --algo_type explicitly from the command line.")

        save_args(args)

    if args.damage_type == "physical":
        args.damage_joint_idx = jnp.array(args.damage_joint_idx)
        args.damage_joint_action = jnp.array(args.damage_joint_action)
        args.zero_sensor_idx = None
    elif args.damage_type == "sensory":
        args.damage_joint_idx = None
        args.damage_joint_action = None
        args.zero_sensor_idx = jnp.array(args.zero_sensor_idx)
    else:
        raise ValueError("Unsupported damage type, please set between physical | sensory")
    return args


if __name__ == "__main__":
    print(jax.devices())

    args = get_args()

    # args.output_path = "./outputs/mapelites_20250701_152736"
    # args.output_path = "./outputs/dcrl_20250703_114735"
    # args.output_path = "./outputs/dcrl_20250710_134938"
    # args.output_path = "./outputs/dcrl_20250704_185243"
    # args.output_path = "outputs/slurm/dcrl_20250710_133450"
    # args.exp_path = args.output_path

    main(
        args.mode,
        args.algo_type,
        args.episode_length, 
        args.seed, 
        args.batch_size, 
        args.num_iterations, 
        args.grid_shape,
        args.policy_hidden_layer_sizes,
        args.damage_joint_idx, 
        args.damage_joint_action,
        args.zero_sensor_idx,
        args.ga_batch_size,
        args.dcrl_batch_size,
        args.ai_batch_size,
        args.lengthscale,
        args.critic_hidden_layer_size,
        args.num_critic_training_steps,
        args.num_pg_training_steps,
        args.replay_buffer_size,
        args.discount,
        args.reward_scaling,
        args.critic_learning_rate,
        args.actor_learning_rate,
        args.policy_learning_rate,
        args.noise_clip,
        args.policy_noise,
        args.soft_tau_update,
        args.policy_delay,
        args.output_path,
        args.exp_path,
        args.env_name,
        args.min_descriptor, 
        args.max_descriptor, 
        args.iso_sigma, 
        args.line_sigma, 
        args.log_period,
        args.max_iters, 
        args.performance_threshold
    )



    