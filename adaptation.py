from typing import Dict
import math
import functools
import time
from tqdm import trange

import jax
import jax.numpy as jnp
import numpy as np

from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.utils.metrics import CSVLogger

import utils.gp_jax as gpx
from utils.new_plot import plot_diff_qd_score, plot_grid_results
from utils.util import log_metrics

from rollout import run_single_rollout, render_rollout_to_html, jit_rollout_fn

def upper_confidence_bound(
    mean, 
    std, 
    kappa=0.05
):
    return jnp.argmax(mean + kappa * std)


def run_online_adaptation(
    env_name,
    repertoire: MapElitesRepertoire,
    env, 
    policy_network, 
    key, 
    exp_path,
    min_descriptor, 
    max_descriptor, 
    grid_shape,
    damage_joint_idx, 
    damage_joint_action, 
    zero_sensor_idx, 
    episode_length, 
    vmin,
    vmax,
    eval_metrics: Dict,
    max_iters=20, 
    performance_threshold=0.9, 
    lengthscale=0.4, 
    noise=1e-3,
    ):


    print("Lengthscale:", lengthscale)
    print("Noise:", noise)
    print("Performance threshold:", performance_threshold)
    print("Max iterations:", max_iters)

    print("damage_joint_idx:", damage_joint_idx)
    print("damage_joint_action:", damage_joint_action)
    print("zero_sensor_idx:", zero_sensor_idx)


    # select the most promising behavior from MAP
    fitnesses = repertoire.fitnesses
    fitnesses = fitnesses.reshape(-1, 1)
        
    next_idx = jnp.argmax(fitnesses)
    print("next_idx: ", next_idx)

    D = gpx.Dataset()
    means_adjusted = jnp.squeeze(fitnesses, axis=1)    # mu_0 = P(x)

    # Define the GP model
    kernel = gpx.kernels.Matern52(lengthscale=lengthscale)
    mean_fn = gpx.mean_functions.Zero()
    prior = gpx.gps.Prior(kernel=kernel, mean_function=mean_fn)
    acquisition_fn = jax.jit(upper_confidence_bound)


    # plot real fitness grid
    # grid_size = math.prod(grid_shape)
    # fitness_rollout_fn = jit_rollout_fn(env, policy_network, episode_length)

    # single_eval = functools.partial(fitness_rollout_fn, 
    #                                damage_joint_idx=damage_joint_idx, 
    #                                damage_joint_action=damage_joint_action,
    #                                zero_sensor_idx=zero_sensor_idx)

    # key, subkey = jax.random.split(key)
    # keys = jax.random.split(subkey, grid_size)
    # batched_rewards = jax.vmap(single_eval)(repertoire.genotypes, keys)    
    # real_repertoire = repertoire.replace(fitnesses=batched_rewards.reshape((-1, 1)))
    # plot_grid_results("real", real_repertoire, min_descriptor, max_descriptor, grid_shape, exp_path, vmin=vmin, vmax=vmax)

    # eval adaptation time
    start_time = time.time()

    for iter_num in trange(max_iters, desc="Adaptation"):
        # input("Press Enter to continue...")
        stop_cond = performance_threshold * jnp.max(means_adjusted)

        if iter_num != 0:
            likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n, obs_stddev=noise)
            posterior = prior * likelihood

            latent_dist = posterior.predict(test_inputs=repertoire.centroids, train_data=D)
            predictive_dist = posterior.likelihood(latent_dist)

            means_residual = predictive_dist.mean()
            stddev = predictive_dist.stddev()
            means_adjusted = jnp.squeeze(fitnesses, axis=1) + means_residual
            next_idx = acquisition_fn(means_adjusted, stddev)
            print(f"\nnext index: {next_idx}\n")
            # print("means_adjusted: ", means_adjusted[filled_mask])
            # filled_mask = jnp.isfinite(means_adjusted)
            # print(f"min estimate: {means_adjusted[filled_mask].min()}")

        next_goal = repertoire.centroids[next_idx]
        eval_metrics["iterative"]["tested_indices"].append(next_idx)
        eval_metrics["iterative"]["tested_behaviours"].append(next_goal)

        # # op1: re-evaluate on the real robot
        params = jax.tree.map(lambda x: x[next_idx], repertoire.genotypes)
        key, subkey = jax.random.split(key)
        rollout = run_single_rollout(env, policy_network, params, subkey, 
                                     damage_joint_idx, damage_joint_action, zero_sensor_idx)
        real_fitness = rollout["rewards"].sum()


        # op2: sample from real fitness grid
        # real_fitness = batched_rewards[next_idx]

        eval_metrics["iterative"]["step_speeds"].append(rollout["rewards"])
        
        obs_dataset = gpx.Dataset(
            X=jnp.expand_dims(next_goal, axis=0),
            y=jnp.expand_dims(real_fitness - jnp.squeeze(fitnesses[next_idx]), axis=[0, 1]),
        )
        D = D + obs_dataset if iter_num != 0 else obs_dataset   # add observation to the dataset

        eval_metrics["iterative"]["tested_fitnesses"].append(real_fitness)
        max_tested_fitness = max(eval_metrics["iterative"]["tested_fitnesses"])
        if real_fitness == max_tested_fitness:
            best_idx = next_idx
        
        # sorted_top_indices = get_top_k_indices(top_k, means_adjusted)

        print(
            f"tested real fitness: {real_fitness:.2f}\n",
            f"tested behaviour: {repertoire.descriptors[next_idx]}\n",
            f"max real fitness by far: {max_tested_fitness:.2f}\n",
            # f"top {top_k} predicted fitness indices: {sorted_top_indices}\n"
        )

        # plot predicted grid after each adaptation iteration
        repertoire = repertoire.replace(fitnesses=jnp.expand_dims(means_adjusted, axis=1))
        plot_grid_results("predicted", repertoire, min_descriptor, max_descriptor, grid_shape, exp_path, iter_num, vmin=vmin, vmax=vmax)

        if (max_tested_fitness >= stop_cond or iter_num == max_iters - 1):

            eval_metrics["global"]["adaptation_time"] = time.time() - start_time
            eval_metrics["global"]["adaptation_steps"] = iter_num
            eval_metrics["global"]["best_tested_index"] = best_idx
            eval_metrics["global"]["best_recovered_behaviour"] = repertoire.descriptors[best_idx]
            eval_metrics["global"]["best_tested_fitness"] = max_tested_fitness

            print(
                f"Adaptation ends in {iter_num} iteration(s).\n",
                f"best index: {best_idx} \n",
                f"Best behaviour after adaptation: {repertoire.descriptors[best_idx]}\n",
            )

            best_params = jax.tree.map(lambda x: x[best_idx], repertoire.genotypes)
            key, subkey = jax.random.split(key)
            rollout = run_single_rollout(env, policy_network, best_params, subkey, 
                                         damage_joint_idx, damage_joint_action, zero_sensor_idx)
            render_rollout_to_html(rollout['states'], env, exp_path + "/post_adaptation_with_damage.html")
            
            real_fitness = rollout['rewards'].sum()
            print(f"real fitness: {real_fitness}")

            print("********adaptation completes********")
            break
    
    # print(f"tested indices: {tested_indices}")
    # print(f"real fitnesses: {real_fitnesses}")
    # print(f"tested goals: {tested_behaviours}")
    return eval_metrics