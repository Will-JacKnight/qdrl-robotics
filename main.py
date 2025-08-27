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

SUPPORTED_EMITTERS = [
    "dcrl",
    # "MAP-Elites"
]



##############
# check device
##############
print(jax.devices())




##############
# parse args
##############
parser = argparse.ArgumentParser(description="ITE Adaptation")
parser.add_argument("--mode", type=str, default="adaptation", help="run mode: training or adaptation (default)")
parser.add_argument("--output_path", type=str, help="relative path to the model")
args, _ = parser.parse_known_args()

if args.mode == "training":
    config_args = load_json(".", "config.json")
else:
    config_args = load_json(args.output_path, "running_args.json")
    config_args["mode"] = "adaptation"

parser.set_defaults(**config_args)

# directory configs
parser.add_argument("--exp_path", type=str, help="relative path to specific damage runs")
parser.add_argument("--algo_type", type=str, default="dcrl", help=f"supported algo: {SUPPORTED_EMITTERS}")

# evaluation step configs
parser.add_argument("--episode_length", type=int, help="Maximum rollout length")
parser.add_argument("--batch_size", type=int, help="Parallel training batch size")
parser.add_argument("--num_iterations", type=int, help="Number of training iterations")

# stochasticity config
parser.add_argument("--dropout-rate", type=int, default=0.2)

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

# Corrected Metrics Configs
parser.add_argument("--log-period", default=10, type=int)
parser.add_argument("--num-reevals", default=4, type=int)
parser.add_argument("--reeval-scan-size", default=0, type=int, help="Not used if 0.")
parser.add_argument("--reeval-fitness-extractor", default="Average", type=str)
parser.add_argument("--reeval-lighter", action="store_true")
parser.add_argument(
    "--reeval-fitness-reproducibility-extractor", default="STD", type=str
)
parser.add_argument("--reeval-descriptor-extractor", default="Average", type=str)
parser.add_argument(
    "--reeval-descriptor-reproducibility-extractor", default="STD", type=str
)

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
if args.mode == "training":
    # args.exp_path = args.output_path
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




##############
# main function
##############

key = jax.random.key(args.seed)
key, subkey = jax.random.split(key)

# map creation
if args.mode == "training":
    match args.algo_type:
        # case "MAP-Elites":
        #     repertoire, metrics = run_map_elites(
        #         env_name, episode_length, policy_hidden_layer_sizes, batch_size, num_iterations, 
        #         grid_shape, min_descriptor, max_descriptor, iso_sigma, line_sigma, log_period, subkey, 
        #         dropout_rate
        #     )
        case "dcrl":
            repertoire, metrics = run_dcrl_map_elites(args=args, key=subkey)
        case _:
            raise ValueError(f"Unknown algo_type: {args.algo_type}")

    save_repertoire_and_metrics(args.output_path, repertoire, metrics)

    env_steps = metrics["iteration"] * args.episode_length * args.batch_size
    plot_map_elites_results(
        env_steps=env_steps, 
        metrics=metrics, 
        repertoire=repertoire,
        min_bd=args.min_descriptor, 
        max_bd=args.max_descriptor, 
        grid_shape=args.grid_shape, 
        output_dir=args.output_path
    )


repertoire, _ = load_repertoire_and_metrics(args.output_path)
env, policy_network, _ = init_env_and_policy_network(
    args.env_name, 
    args.episode_length,
    args.policy_hidden_layer_sizes, 
    args.dropout_rate
)

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

if args.mode == "training":
    key, subkey = jax.random.split(key)
    rollout = run_single_rollout(env, policy_network, params, subkey)
    render_rollout_to_html(rollout['states'], env, args.output_path + "/pre_adaptation_without_damage.html")
else:
    key, subkey = jax.random.split(key)
    rollout = run_single_rollout(env, policy_network, params, subkey,
                                args.damage_joint_idx, args.damage_joint_action, args.zero_sensor_idx)
    render_rollout_to_html(rollout['states'], env, args.exp_path + "/pre_adaptation_with_damage.html")

    key, subkey = jax.random.split(key)
    run_online_adaptation(
        env_name=args.env_name, 
        repertoire=repertoire, 
        env=env, 
        policy_network=policy_network, 
        key=subkey, 
        exp_path=args.exp_path, 
        min_descriptor=args.min_descriptor, 
        max_descriptor=args.max_descriptor, 
        grid_shape=args.grid_shape, 
        damage_joint_idx=args.damage_joint_idx, 
        damage_joint_action=args.damage_joint_action, 
        zero_sensor_idx=args.zero_sensor_idx,
        episode_length=args.episode_length, 
        max_iters=args.max_iters, 
        performance_threshold=args.performance_threshold,
    )