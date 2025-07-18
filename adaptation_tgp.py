from typing import Dict

# import utils.gp_jax as gpx
import tinygp as tgp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from tqdm import trange
from utils.new_plot import plot_live_grid_update


from rollout import run_single_rollout


def upper_confidence_bound(mean: jnp.ndarray, std: jnp.ndarray, kappa=0.05) -> jnp.ndarray:
    return jnp.argmax(mean + kappa * std)
    

def run_online_adaptation(
                      repertoire: MapElitesRepertoire,
                      env, policy_network, key, output_path,
                      damage_joint_idx, damage_joint_action,
                      max_iters=20, performance_threshold=0.9, lengthscale=0.4, noise=1e-3):


    print("Lengthscale:", lengthscale)
    print("Noise:", noise)
    print("Performance threshold:", performance_threshold)
    print("Max iterations:", max_iters)

    # select the most promising behavior from MAP
    fitnesses = repertoire.fitnesses
    next_idx = jnp.argmax(fitnesses)

    tested_indices = []
    real_fitnesses = []
    tested_goals = []
    means_adjusted = fitnesses    # mu_0 = P(x)

    # Define the GP model
    kernel = tgp.kernels.Matern52(scale=lengthscale)
    mean_fn = lambda x: jnp.zeros_like(x[..., 0])  # Zero mean function
    acquisition_fn = jax.jit(upper_confidence_bound)

    for iter_num in trange(max_iters, desc="Adaptation"):
        # input("Press Enter to continue...")
        stop_cond = performance_threshold * jnp.max(means_adjusted)

        if iter_num != 0:
            # Create a new GP with current observations
            gp = tgp.GaussianProcess(
                kernel,
                X_obs,
                diag=jnp.broadcast_to(noise, y_obs.shape[0]),
                mean=mean_fn
            )
            # breakpoint()
            means_residual, variances = gp.predict(y_obs, X_test=repertoire.centroids, return_var=True)
            means_adjusted = jnp.squeeze(fitnesses, axis=1) + means_residual         ## mismatching shape
            next_idx = acquisition_fn(means_adjusted, jnp.sqrt(variances))
            print(f"next_idx: {next_idx}")

        next_goal = repertoire.centroids[next_idx]
        tested_indices.append(next_idx)
        tested_goals.append(next_goal)

        # evaluate on the real robot
        params = jax.tree.map(lambda x: x[next_idx], repertoire.genotypes)
        key, subkey = jax.random.split(key)
        rollout = run_single_rollout(env, policy_network, params, subkey, damage_joint_idx, damage_joint_action, None)          # rollout = {'rewards': jnp.array, 'state': jnp.array}
        real_fitness = rollout['rewards'].sum()

        # Add new observation
        if iter_num == 0:
            X_obs = jnp.expand_dims(next_goal, axis=0)
            y_obs = jnp.array([real_fitness - fitnesses[next_idx]]).reshape(-1)
        else:
            X_obs = jnp.concatenate([X_obs, jnp.expand_dims(next_goal, axis=0)], axis=0)
            y_obs = jnp.concatenate([y_obs, jnp.array([real_fitness - fitnesses[next_idx]]).reshape(-1)], axis=0)

        real_fitnesses.append(real_fitness.item())
        max_tested_fitness = max(real_fitnesses)
        if real_fitness.item() == max_tested_fitness:
            best_idx = next_idx

        print(
            f"real fitness: {real_fitness:.2f}\n",
            f"tested behaviour: {repertoire.descriptors[next_idx]}\n",
            f"Max real fitness by far: {max_tested_fitness:.2f}\n",
        )
        
        # save live plots after each adaptation
        repertoire = repertoire.replace(fitnesses=means_adjusted)
        plot_live_grid_update(iter_num, repertoire, min_descriptor, max_descriptor, grid_shape, output_path)

        if (max_tested_fitness >= stop_cond or iter_num == max_iters - 1):
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
                                         output_path + "/post_adaptation_with_damage.html")
            
            real_fitness = rollout['rewards'].sum()
            print(f"real fitness: {real_fitness}")
            # breakpoint()

            break
    
    # print(f"tested indices: {tested_indices}")
    # print(f"real fitnesses: {real_fitnesses}")
    # print(f"tested goals: {tested_goals}")

    return np.array(tested_indices), np.array(real_fitnesses), np.array(tested_goals)