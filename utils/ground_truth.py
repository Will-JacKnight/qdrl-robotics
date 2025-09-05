from typing import Tuple
import math
import functools

import jax
import jax.numpy as jnp

from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.tasks.brax.v1.env_creators import scoring_function_brax_envs
from qdax.tasks.brax.v1 import descriptor_extractor
from qdax.custom_types import RNGKey

from rollout import play_damage_step_fn

# obsolete
def eval_real_fitness(
    env_name: str,
    env,
    policy_network,
    episode_length: int,
    repertoire: MapElitesRepertoire,
    grid_shape: Tuple[int, ...],
    key: RNGKey,
    damage_joint_idx: jnp.ndarray,
    damage_joint_action: jnp.ndarray,
    zero_sensor_idx: jnp.ndarray,
) -> MapElitesRepertoire:

    grid_size = math.prod(grid_shape)

    reset_fn = jax.jit(env.reset)
    descriptor_extraction_fn = descriptor_extractor[env_name]

    damage_step_fn = functools.partial(play_damage_step_fn,
                                    env=env,
                                    policy_network=policy_network,
                                    damage_joint_idx=damage_joint_idx, 
                                    damage_joint_action=damage_joint_action,
                                    zero_sensor_idx=zero_sensor_idx)

    scoring_fn = functools.partial(
        scoring_function_brax_envs,
        episode_length=episode_length,
        play_reset_fn=reset_fn,
        play_step_fn=damage_step_fn,
        descriptor_extractor=descriptor_extraction_fn,
    )

    # parallel evals of all genotypes in the grid
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, grid_size)
    breakpoint()
    (fitnesses, descriptors, extra_scores) = jax.vmap(scoring_fn)(repertoire.genotypes, keys)

    real_repertoire = repertoire.replace(
        fitnesses=fitnesses,
        descriptors=descriptors,
    )
    return real_repertoire
