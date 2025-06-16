import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import gpjax as gpx
from tqdm import trange
from typing import Dict

from rollout import run_single_rollout
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire

def upper_confidence_bound(mean: jnp.array, std: jnp.array, kappa=0.05) -> jnp.array:
    return jnp.argmax(mean + kappa * std)
    

def run_online_adaptation(
                      repertoire: MapElitesRepertoire,
                      env, policy_network, key,
                      max_iters=20, performance_threshold=0.9, lengthscale=0.4, noise=1e-3):


    print("Lengthscale:", lengthscale)
    print("Noise:", noise)
    print("Performance threshold:", performance_threshold)
    print("Max iterations:", max_iters)
    # rollouts = {
    #     'index1': {'reward': jnp.array, 'state': jnp.array},
    #     'index2': {'reward': jnp.array, 'state': jnp.array},
    # }

    # fitnesses = jnp.array([rollout['rewards'].sum() for rollout in rollouts.values()])
    # next_idx = list(rollouts.keys())[jnp.argmax(fitnesses)]   # pick the rollout with highest fitness

    # select the most promising behavior from MAP
    fitnesses = repertoire.fitnesses
    next_idx = jnp.argmax(fitnesses)

    D = gpx.Dataset()
    tested_indices = []
    real_fitnesses = []
    # tested_goals = []
    means_adjusted = fitnesses    # mu_0 = P(x)

    # Define the GP model
    kernel = gpx.kernels.Matern52(lengthscale=lengthscale)
    mean_fn = gpx.mean_functions.Zero()
    prior = gpx.gps.Prior(kernel=kernel, mean_function=mean_fn)
    acquisition_fn = jax.jit(upper_confidence_bound)


    for iter_num in trange(max_iters, desc="Adaptation"):
        input("Press Enter to continue...")
        stop_cond = performance_threshold * jnp.max(means_adjusted)

        if iter_num != 0:
            likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n, obs_stddev=noise)
            posterior = prior * likelihood
            latent_dist = posterior.predict(test_inputs=next_idx, train_data=D)
            predictive_dist = posterior.likelihood(latent_dist)

            means_residual = predictive_dist.mean()
            variances = predictive_dist.variance()

            means_adjusted = fitnesses + means_residual
            next_idx = acquisition_fn(means_adjusted, variances)

        # next_goal = goals[next_idx]
        tested_indices.append(next_idx)
        # tested_goals.append(next_goal)

        # evaluate on the real robot
        params = jax.tree.map(lambda x: x[next_idx], repertoire.genotypes)
        rollout = run_single_rollout(env, policy_network, params, key)          # rollout = {'rewards': jnp.array, 'state': jnp.array}
        real_fitness = rollout['rewards'].sum()

        obs_dataset = gpx.Dataset(
            X=jnp.expand_dims(next_idx, axis=0),
            y=jnp.expand_dims(real_fitness - fitnesses[next_idx], axis=[0, 1])
        )
        D = D + obs_dataset if iter_num != 0 else obs_dataset   # add observation to the dataset

        real_fitnesses.append(real_fitness.item())
        max_tested_fitness = max(real_fitnesses)

        if max_tested_fitness >= stop_cond:
            # print(f"Early stopping: fitness {max_tested_fitness:.3f} >= threshold {stop_cond:.3f}")
            break

    # return np.array(tested_indices), np.array(real_fitnesses)