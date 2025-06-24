from typing import Dict

import gpjax as gpx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from tqdm import trange

from rollout import run_single_rollout


def upper_confidence_bound(mean: jnp.array, std: jnp.array, kappa=0.05) -> jnp.array:
    return jnp.argmax(mean + kappa * std)
    

def run_online_adaptation(
                      repertoire: MapElitesRepertoire,
                      env, policy_network, key,
                      damage_joint_idx, damage_joint_action,
                      max_iters=20, performance_threshold=0.9, lengthscale=0.4, noise=1e-3):


    print("Lengthscale:", lengthscale)
    print("Noise:", noise)
    print("Performance threshold:", performance_threshold)
    print("Max iterations:", max_iters)

    # select the most promising behavior from MAP
    fitnesses = repertoire.fitnesses
    next_idx = jnp.argmax(fitnesses)

    D = gpx.Dataset()
    tested_indices = []
    real_fitnesses = []
    tested_goals = []
    means_adjusted = fitnesses    # mu_0 = P(x)

    # Define the GP model
    kernel = gpx.kernels.Matern52(lengthscale=lengthscale)
    mean_fn = gpx.mean_functions.Zero()
    prior = gpx.gps.Prior(kernel=kernel, mean_function=mean_fn)
    acquisition_fn = jax.jit(upper_confidence_bound)


    for iter_num in trange(max_iters, desc="Adaptation"):
        # input("Press Enter to continue...")
        stop_cond = performance_threshold * jnp.max(means_adjusted)

        if iter_num != 0:
            likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n, obs_stddev=noise)
            posterior = prior * likelihood

            latent_dist = posterior.predict(test_inputs=repertoire.centroids, train_data=D)
            predictive_dist = posterior.likelihood(latent_dist)

            means_residual = predictive_dist.mean()
            variances = predictive_dist.variance()

            means_adjusted = fitnesses + means_residual
            next_idx = acquisition_fn(means_adjusted, variances)

        next_goal = repertoire.centroids[next_idx]
        tested_indices.append(next_idx)
        tested_goals.append(next_goal)

        # evaluate on the real robot
        params = jax.tree.map(lambda x: x[next_idx], repertoire.genotypes)
        key, subkey = jax.random.split(key)
        rollout = run_single_rollout(env, policy_network, params, subkey, damage_joint_idx, damage_joint_action, None)          # rollout = {'rewards': jnp.array, 'state': jnp.array}
        real_fitness = rollout['rewards'].sum()

        obs_dataset = gpx.Dataset(
            X=jnp.expand_dims(next_goal, axis=0),
            y=jnp.expand_dims(real_fitness - fitnesses[next_idx], axis=[0, 1]),
        )
        D = D + obs_dataset if iter_num != 0 else obs_dataset   # add observation to the dataset

        real_fitnesses.append(real_fitness.item())
        max_tested_fitness = max(real_fitnesses)
        if real_fitness.item() == max_tested_fitness:
            best_idx = next_idx

        print(
            f"real fitness: {real_fitness:.2f}\n",
            f"tested behaviour: {repertoire.descriptors[next_idx]}\n",
            f"Max real fitness by far: {max_tested_fitness:.2f}\n",
        )

        if max_tested_fitness >= stop_cond:
            # print(f"Early stopping: fitness {max_tested_fitness:.3f} >= threshold {stop_cond:.3f}")
            print(
                f"Adaptation ends in {iter_num} iteration(s).\n",
                f"best index: {best_idx} \n",
                f"Best behaviour after adaptation: {repertoire.descriptors[best_idx]}\n",
            )

            best_params = jax.tree.map(lambda x: x[best_idx], repertoire.genotypes)
            key, subkey = jax.random.split(key)
            rollout = run_single_rollout(env, policy_network, best_params, subkey, 
                                         damage_joint_idx, damage_joint_action,
                                         "./outputs/post_adaptation_with_damage.html")
            
            real_fitness = rollout['rewards'].sum()
            print(f"real fitness: {real_fitness}")
            breakpoint()

            break
    
    # print(f"tested indices: {tested_indices}")
    # print(f"real fitnesses: {real_fitnesses}")
    # print(f"tested goals: {tested_goals}")

    return np.array(tested_indices), np.array(real_fitnesses), np.array(tested_goals)