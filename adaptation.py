from typing import Dict
import math
import functools

import utils.gp_jax as gpx
# import gpjax as gpx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from tqdm import trange
from utils.new_plot import plot_diff_qd_score, plot_grid_results

from rollout import run_single_rollout, render_rollout_to_html, create_jit_rollout_fn


def run_online_adaptation(
    repertoire: MapElitesRepertoire,
    env, policy_network, key, exp_path,
    min_descriptor, max_descriptor, grid_shape,
    damage_joint_idx, damage_joint_action, zero_sensor_idx, 
    episode_length, max_iters=20, performance_threshold=0.9, 
    lengthscale=0.4, noise=1e-3
):


    print("Lengthscale:", lengthscale)
    print("Noise:", noise)
    print("Performance threshold:", performance_threshold)
    print("Max iterations:", max_iters)

    # select the most promising behavior from MAP
    fitnesses = repertoire.fitnesses
    next_idx = jnp.argmax(fitnesses)

    D = gpx.Dataset()
    tested_indices = []
    real_iter_fitnesses = []
    tested_goals = []
    means_adjusted = jnp.squeeze(fitnesses, axis=1)    # mu_0 = P(x)
    
    def upper_confidence_bound(mean, std, kappa=0.05):
        return jnp.argmax(mean + kappa * std)

    # Define the GP model
    kernel = gpx.kernels.Matern52(lengthscale=lengthscale)
    mean_fn = gpx.mean_functions.Zero()
    prior = gpx.gps.Prior(kernel=kernel, mean_function=mean_fn)
    acquisition_fn = jax.jit(upper_confidence_bound)


    # plot real fitness grid
    avg_diff_qd_scores = []
    grid_size = math.prod(grid_shape)
    jit_rollout_fn = create_jit_rollout_fn(env, policy_network, episode_length)

    single_eval = functools.partial(jit_rollout_fn, 
                                   damage_joint_idx=damage_joint_idx, 
                                   damage_joint_action=damage_joint_action)

    # batched_rewards = []
    # for i in trange(grid_size):
    #     key, subkey = jax.random.split(key)
    #     params = jax.tree.map(lambda x: x[i], repertoire.genotypes)
    #     batched_rewards.append(single_eval(params, subkey))

    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, grid_size)
    batched_rewards = jax.vmap(single_eval)(repertoire.genotypes, keys)

    repertoire = repertoire.replace(fitnesses=batched_rewards.reshape((-1, 1)))

    plot_grid_results("real", repertoire, min_descriptor, max_descriptor, grid_shape, exp_path)


    top_k = 10
    indices = jnp.argpartition(batched_rewards, -top_k)[-top_k:]     # get indices of top 10 fitness behaviours
    sorted_top_indices = indices[jnp.argsort(-batched_rewards[indices])] # behaviour indices in descending fitnesses
    
    best_real_idx = jnp.argmax(batched_rewards)
    print(f"\nbest index after damage: {best_real_idx}")
    print(f"best real behaviour after damage: {repertoire.descriptors[best_real_idx]}")
    print(f"best real fitness after damage:{jnp.max(batched_rewards):.2f}")
    print(f"top {top_k} real fitness indices: {sorted_top_indices} \n")

    best_params = jax.tree.map(lambda x: x[best_real_idx], repertoire.genotypes)
    key, subkey = jax.random.split(key)
    rollout = run_single_rollout(env, policy_network, best_params, subkey, 
                                 damage_joint_idx, damage_joint_action, zero_sensor_idx)
    render_rollout_to_html(rollout['states'], env, exp_path + "/best_real_fitness.html")


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
            print(f"next index: {next_idx}")

        next_goal = repertoire.centroids[next_idx]
        tested_indices.append(next_idx)
        tested_goals.append(next_goal)

        # evaluate on the real robot
        params = jax.tree.map(lambda x: x[next_idx], repertoire.genotypes)
        key, subkey = jax.random.split(key)
        rollout = run_single_rollout(env, policy_network, params, subkey, 
                                     damage_joint_idx, damage_joint_action, zero_sensor_idx)
        real_fitness = rollout['rewards'].sum()

        obs_dataset = gpx.Dataset(
            X=jnp.expand_dims(next_goal, axis=0),
            y=jnp.expand_dims(real_fitness - fitnesses[next_idx], axis=[0, 1]),
        )
        D = D + obs_dataset if iter_num != 0 else obs_dataset   # add observation to the dataset

        real_iter_fitnesses.append(real_fitness)
        max_tested_fitness = max(real_iter_fitnesses)
        if real_fitness == max_tested_fitness:
            best_idx = next_idx

        print(
            f"real fitness: {real_fitness:.2f}\n",
            f"tested behaviour: {repertoire.descriptors[next_idx]}\n",
            f"Max real fitness by far: {max_tested_fitness:.2f}\n",
        )

        # plot predicted grid after each adaptation iteration
        repertoire = repertoire.replace(fitnesses=jnp.expand_dims(means_adjusted, axis=1))
        plot_grid_results("predicted", repertoire, min_descriptor, max_descriptor, grid_shape, exp_path, iter_num)

        # calulate diff qd score
        diff_score = np.zeros(grid_size)
        filled_mask = jnp.isfinite(means_adjusted) & jnp.isfinite(batched_rewards)
        diff_score[filled_mask] = (batched_rewards[filled_mask] - means_adjusted[filled_mask])
        avg_diff_qd_score = jnp.abs(diff_score.sum()) / jnp.sum(filled_mask)

        print(f"diff QD score: {avg_diff_qd_score:.3f}\n")
        avg_diff_qd_scores.append(avg_diff_qd_score)

        if (max_tested_fitness >= stop_cond or iter_num == max_iters - 1):

            adaptation_steps = iter_num + 1
            # print(f"Early stopping: fitness {max_tested_fitness:.3f} >= threshold {stop_cond:.3f}")
            print(
                f"Adaptation ends in {iter_num + 1} iteration(s).\n",
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

            # print(f"survive reward: {rollout['survive_reward'].sum()}")
            # print(f"forward reward: {rollout['forward_reward'].sum()}")
            # print(f"control reward: {rollout['control_reward'].sum()}")
            # print(f"contact reward: {rollout['contact_reward'].sum()}")
            
            print("********adaptation completes********")
            break
    
    plot_diff_qd_score(adaptation_steps, avg_diff_qd_scores, exp_path)
    # print(f"tested indices: {tested_indices}")
    # print(f"real fitnesses: {real_fitnesses}")
    # print(f"tested goals: {tested_goals}")

    # return np.array(tested_indices), np.array(real_iter_fitnesses), np.array(tested_goals)