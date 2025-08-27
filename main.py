from typing import Any, Dict, Iterable, List, Optional, Tuple, Literal
import argparse
import sys

import jax
import jax.numpy as jnp

from adaptation import run_online_adaptation
from map_elites import run_map_elites
from dcrl import run_dcrl_map_elites
from rollout import run_single_rollout, init_env_and_policy_network, render_rollout_to_html
from utils.util import load_json,load_repertoire_and_metrics, save_repertoire_and_metrics, save_args
from utils.new_plot import plot_map_elites_results
from setup_containers import get_evals_per_offspring, get_batch_size, get_sampling_size

SUPPORTED_CONTAINERS = [
    "MAP-Elites_Sampling",
    "Archive-Sampling",
    "Extract-MAP-Elites",
]

SUPPORTED_DAMAGES = [
    "physical",
    "sensory",
]

def main(
    mode: Literal["training", "adaptation"],
    container: str,
    algo_type: str,
    episode_length: int, 
    seed: int, 
    batch_size: int, 
    num_iterations: int, 
    grid_shape: Tuple[int, ...],
    policy_hidden_layer_sizes: Tuple[int, ...],
    damage_joint_idx: jnp.ndarray, 
    damage_joint_action: jnp.ndarray,
    zero_sensor_idx: jnp.ndarray, 
    ga_batch_size: int,
    dcrl_batch_size: int,
    ai_batch_size: int,
    lengthscale: float,
    critic_hidden_layer_size: Tuple[int, ...],
    num_critic_training_steps: int,
    num_pg_training_steps: int,
    replay_buffer_size: int,
    discount: float,
    reward_scaling: float,
    critic_learning_rate: float,
    actor_learning_rate: float,
    policy_learning_rate: float,
    noise_clip: float,
    policy_noise: float,
    soft_tau_update: float,
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
    performance_threshold,
    dropout_rate: float,
    num_samples: int,
    depth: int,
    max_number_evals: int,
    fitness_extractor: str,
    fitness_reproducibility_extractor: str,
    descriptor_extractor: str,
    descriptor_reproducibility_extractor: str,
    as_repertoire_num_samples: int,
    extract_type: str,
    emit_batch_size: int,
):
    

    key = jax.random.key(seed)
    key, subkey = jax.random.split(key)


    # map creation
    if mode == "training":
        match algo_type:
            case "mapelites":
                repertoire, metrics = run_map_elites(
                    env_name, episode_length, policy_hidden_layer_sizes, batch_size, num_iterations, 
                    grid_shape, min_descriptor, max_descriptor, iso_sigma, line_sigma, log_period, subkey, 
                    dropout_rate
                )
            case "dcrl":
                repertoire, metrics = run_dcrl_map_elites(
                    env_name, container, episode_length, policy_hidden_layer_sizes, batch_size, num_iterations, 
                    grid_shape, min_descriptor, max_descriptor, iso_sigma, line_sigma, ga_batch_size, 
                    dcrl_batch_size, ai_batch_size, lengthscale, critic_hidden_layer_size, num_critic_training_steps,
                    num_pg_training_steps, replay_buffer_size, discount, reward_scaling, critic_learning_rate,
                    actor_learning_rate, policy_learning_rate, noise_clip, policy_noise, soft_tau_update,
                    policy_delay, log_period, subkey, dropout_rate, 
                    num_samples, depth, max_number_evals, fitness_extractor, fitness_reproducibility_extractor, 
                    descriptor_extractor, descriptor_reproducibility_extractor,
                    as_repertoire_num_samples, extract_type, emit_batch_size,
                )
            case _:
                raise ValueError(f"Unknown algo_type: {algo_type}")

        save_repertoire_and_metrics(output_path, repertoire, metrics)

        env_steps = metrics["iteration"] * episode_length * batch_size
        plot_map_elites_results(env_steps=env_steps, metrics=metrics, repertoire=repertoire, 
                                min_bd=min_descriptor, max_bd=max_descriptor, grid_shape=grid_shape, output_dir=output_path)


    repertoire, _ = load_repertoire_and_metrics(output_path)
    env, policy_network, _ = init_env_and_policy_network(env_name, episode_length, policy_hidden_layer_sizes, dropout_rate)
 
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
    else:
        key, subkey = jax.random.split(key)
        rollout = run_single_rollout(env, policy_network, params, subkey,
                                    damage_joint_idx, damage_joint_action, zero_sensor_idx)
        render_rollout_to_html(rollout['states'], env, exp_path + "/pre_adaptation_with_damage.html")

        key, subkey = jax.random.split(key)
        run_online_adaptation(env_name, repertoire, env, policy_network, subkey, exp_path, 
                            min_descriptor, max_descriptor, grid_shape, 
                            damage_joint_idx, damage_joint_action, zero_sensor_idx,
                            episode_length, max_iters, performance_threshold)



def get_args():
    parser = argparse.ArgumentParser(description="ITE Adaptation")
    parser.add_argument("--mode", type=str, default="adaptation", help="run mode: training or adaptation (default)")
    parser.add_argument("--output_path", type=str, help="relative path to the model")
    args, _ = parser.parse_known_args()

    if args.mode == "training":
        config_args = load_json(".", "config.json")
    else:
        config_args = load_json(args.output_path, "running_args.json")

    parser.set_defaults(**config_args)
    
    # directory configs
    parser.add_argument("--exp_path", type=str, help="relative path to specific damage runs")
    parser.add_argument("--algo_type", type=str, help="choose from: mapelites || dcrl")

    # evaluation step configs
    parser.add_argument("--episode_length", type=int, help="Maximum rollout length")
    parser.add_argument("--batch_size", type=int, help="Parallel training batch size")
    parser.add_argument("--num_iterations", type=int, help="Number of training iterations")

    # UQD configs
    parser.add_argument("--container", type=str, help=f"supported containers: {SUPPORTED_CONTAINERS}")
    parser.add_argument("--num-samples", default=1, type=int, help="number of first evaluations per genotype")
    parser.add_argument("--sampling-size", default=4096, type=int, help="number of evaluations per generation")
    parser.add_argument("--depth", default=1, type=int)
    parser.add_argument("--fitness-extractor", default="Average", type=str)
    parser.add_argument("--fitness-reproducibility-extractor", default="STD", type=str)
    parser.add_argument("--descriptor-extractor", default="Average", type=str)
    parser.add_argument("--descriptor-reproducibility-extractor", default="STD", type=str)
    parser.add_argument(
        "--max-number-evals", 
        default=10, 
        type=int, 
        help="capacity of resamples stored for each genotype (jax-compatible)"
    )

    # archive sampling configs
    parser.add_argument(
        "--as-repertoire-num-samples", 
        default=1, 
        type=int,
        help="number of re-evaluations of extracted genotype from repertoire"
    )

    # extract-QD configs
    parser.add_argument("--extract-type", default="proportional", type=str)
    parser.add_argument("--extract-proportion-resample", default=0.25, type=float)
    parser.add_argument("--extract-cap-resample", default=2048, type=int)

    # damage configs
    parser.add_argument("--damage_type", type=str, help=f"Damage type: {SUPPORTED_DAMAGES}")
    parser.add_argument("--damage_joint_idx", type=int, nargs='+', help="Index of the damaged joint")
    parser.add_argument("--damage_joint_action", type=float, nargs='+', help="Action value of the damaged joint")
    parser.add_argument("--zero_sensor_idx", type=int, nargs='+', help="Index of the zero sensor")

    args = parser.parse_args()

    evals_per_offspring = get_evals_per_offspring(args=args)
    if args.sampling_size != 0:
        # Compute batch_size from sampling_size
        (
            args.batch_size,
            args.init_batch_size,
            args.emit_batch_size,
            args.real_evals_per_iter,
        ) = get_batch_size(
            sampling_size=args.sampling_size,
            evals_per_offspring=evals_per_offspring,
            args=args,
        )
    else:
        # Compute sampling_size from batch_size
        (
            args.batch_size,
            args.init_batch_size,
            args.emit_batch_size,
            args.real_evals_per_iter,
        ) = get_sampling_size(
            batch_size=args.batch_size,
            evals_per_offspring=evals_per_offspring,
            args=args,
        )

    print(f"Using batch-size: {args.batch_size} and sampling-size: {args.sampling_size}.")
    print(f"With real_evals_per_iter: {args.real_evals_per_iter}")


    # format transformation
    args.grid_shape = tuple(args.grid_shape)
    args.policy_hidden_layer_sizes = tuple(args.policy_hidden_layer_sizes)
    args.critic_hidden_layer_size = tuple(args.critic_hidden_layer_size)

    # value check
    if len(args.damage_joint_idx) != len(args.damage_joint_action):
        raise ValueError("Number of damage joint actions need to match the number of damage joint indices.")

    # save args in training mode
    print(f"algo type: {args.algo_type}")
    print(f"Running on: {args.mode} mode")
    if args.mode == "training":
        # args.exp_path = args.output_path

        if "--algo_type" not in sys.argv:
            raise ValueError("You must specify --algo_type explicitly from the command line.")

        if args.container not in SUPPORTED_CONTAINERS:
            raise ValueError(f"container currently not supported, choose between: {SUPPORTED_CONTAINERS}")

        assert args.as_repertoire_num_samples > 0, "!!!ERROR!!! Invalid repertoire_num_samples."

        save_args(args, "running_args.json")

    if args.damage_type == "physical":
        args.damage_joint_idx = jnp.array(args.damage_joint_idx)
        args.damage_joint_action = jnp.array(args.damage_joint_action)
        args.zero_sensor_idx = jnp.array([], dtype=jnp.int32)
    elif args.damage_type == "sensory":
        args.damage_joint_idx = jnp.array([], dtype=jnp.int32)
        args.damage_joint_action = jnp.array([], dtype=jnp.float32)
        args.zero_sensor_idx = jnp.array(args.zero_sensor_idx)
    else:
        raise ValueError("Unsupported damage type, please set between physical | sensory")

    return args


if __name__ == "__main__":
    print(jax.devices())

    args = get_args()

    main(
        args.mode,
        args.container,
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
        args.performance_threshold,
        args.dropout_rate,
        args.num_samples,
        args.depth,
        args.max_number_evals,
        args.fitness_extractor,
        args.fitness_reproducibility_extractor,
        args.descriptor_extractor,
        args.descriptor_reproducibility_extractor,
        args.as_repertoire_num_samples,
        args.extract_type,
        args.emit_batch_size,
    )
