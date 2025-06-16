# File: ite_algorithm.py
# Authored by Lisa

import jax.numpy as jnp
import jax
import numpy as np
import gpjax as gpx
from tqdm import trange
from ite.rollout import run_single_rollout
from ite.acquisition import upper_confidence_bound


def run_ite_adaptation(goals, rollouts, env, agent, goal_sampler, episode_length, damaged_joint_idx, damaged_joint_value, is_unsupervised,
                       use_vf, max_iters=20, performance_threshold=0.9, lengthscale=0.4, noise=1e-3):
    
    print("Lengthscale:", lengthscale)
    print("Noise:", noise)
    print("Performance threshold:", performance_threshold)
    print("Max iterations:", max_iters)
    
    # Simulation predictions
    if use_vf:
        fitnesses = jnp.array([rollout['values'] for rollout in rollouts.values()])
    else:
        fitnesses = jnp.array([rollout['rewards'].sum() for rollout in rollouts.values()])
    
    next_idx = list(rollouts.keys())[jnp.argmax(fitnesses)]
    print("Current best fitness:", fitnesses[next_idx])
    print("Current best goal:", goals[next_idx])
    print("Current best goal idx:", next_idx)

    # Observation data -- initialise to be empty
    D = gpx.Dataset()
    tested_indices = []
    real_fitnesses = []
    tested_goals = []
    means_adjusted = fitnesses

    # Define the GP model
    kernel = gpx.kernels.Matern52(lengthscale=lengthscale)
    meanf = gpx.mean_functions.Zero()
    prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
    acquisition_function = jax.jit(upper_confidence_bound)

    for iter_num in trange(max_iters, desc="ITE adaptation"):

        input("Press Enter to continue...")
        stop_cond = performance_threshold * jnp.max(means_adjusted)

        if iter_num != 0:
            # Update the GP model with the new observations
            likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n, obs_stddev=noise)
            posterior = prior * likelihood
            latent_dist = posterior.predict(goals, train_data=D)
            predictive_dist = posterior.likelihood(latent_dist)

            # Predict means and variances for diff. between 
            # sim. performance and real performance
            means_residual = predictive_dist.mean()
            variances = predictive_dist.stddev()
            # Adjust means by adding back sim. performance
            means_adjusted = fitnesses + means_residual

            # Apply acquisition function and select next test goal
            next_idx = int(acquisition_function(means_adjusted, variances))

        next_goal = goals[next_idx] 
        tested_indices.append(next_idx)
        tested_goals.append(next_goal)
        
        # Evaluate the goal on the robot
        # run_single_rollout resets the environment at the start of each rollout 
        rollout = run_single_rollout(env, agent, next_goal, goal_sampler.encode, episode_length, damaged_joint_idx, damaged_joint_value, is_unsupervised)
        real_fitness = rollout['rewards'].sum()
        print("Real fitness:", real_fitness)
        print("Simulated fitness:", fitnesses[next_idx])
        print("Tested goal:", next_goal)
        print("Tested goal idx:", next_idx)

        # Update the dataset with the new observation
        obs_dataset = gpx.Dataset(
            X=jnp.expand_dims(next_goal, axis=0), 
            y=jnp.expand_dims(real_fitness-fitnesses[next_idx], axis=[0, 1])
        )
        D = D + obs_dataset if iter_num != 0 else obs_dataset
        
        # Compute max performance
        real_fitnesses.append(real_fitness.item())
        max_tested_fitness = max(real_fitnesses)
        print("Max. real fitness at iteration", iter_num, ":", max_tested_fitness)
        print("Stop condition:", performance_threshold * jnp.max(means_adjusted))
        print("Stop condition value:", stop_cond)

        # Check for convergence
        if max_tested_fitness >= stop_cond:
            print(f"Early stopping: fitness {max_tested_fitness:.3f} >= threshold {stop_cond:.3f}")
            break

    return np.array(tested_indices), np.array(real_fitnesses), np.array(tested_goals)
