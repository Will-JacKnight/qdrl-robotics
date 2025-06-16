import jax
import functools
import time
import jax.numpy as jnp
import matplotlib.pyplot as plt

from qdax.core.map_elites import MAPElites
from qdax.core.containers.mapelites_repertoire import compute_euclidean_centroids
from qdax.tasks.arm import arm_scoring_function
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.utils.metrics import default_qd_metrics
from qdax.utils.plotting import plot_map_elites_results

seed = 42
num_param_dimensions = 100  # num DoF arm
init_batch_size = 100           # num of randomly generated solutions to initialise the repertoire
batch_size = 1024               # training batch for parallelisation
num_iterations = 50
grid_shape = (100, 100)
min_param = 0.0
max_param = 1.0
min_descriptor = 0.0
max_descriptor = 1.0

# Init a random key
key = jax.random.key(seed)
# Init population of controllers
key, subkey = jax.random.split(key)

init_variables = jax.random.uniform(
    subkey,
    shape=(init_batch_size, num_param_dimensions),
    minval=min_param,
    maxval=max_param,
)

# Define emitter
variation_fn = functools.partial(
    isoline_variation,
    iso_sigma=0.05,         # isotropic Gaussian noise: mutation-like (explore the vicinity of existing elites)
    line_sigma=0.1,         # directional Gaussian noise: line-based, crossover-like (explore the vector connecting 2 elite solutions)
    minval=min_param,
    maxval=max_param,
)
mixing_emitter = MixingEmitter(
    mutation_fn=lambda x, y: (x, y),
    variation_fn=variation_fn,
    variation_percentage=1.0,
    batch_size=batch_size,
)

# Define a metrics function
metrics_fn = functools.partial(
    default_qd_metrics,
    qd_offset=0.0,              # ps. only usded for QD score computation
)

# Instantiate MAP-Elites
map_elites = MAPElites(
    scoring_function=arm_scoring_function,
    emitter=mixing_emitter,
    metrics_function=metrics_fn,
)

# Compute the centroids
centroids = compute_euclidean_centroids(
    grid_shape=grid_shape,
    minval=min_descriptor,
    maxval=max_descriptor,
)
# Initializes repertoire and emitter state
key, subkey = jax.random.split(key)
repertoire, emitter_state, init_metrics = map_elites.init(init_variables, centroids, subkey)

metrics = {key: jnp.array([]) for key in ["iteration", "qd_score", "coverage", "max_fitness", "time"]}

# Set up init metrics
init_metrics = jax.tree.map(lambda x: jnp.array([x]) if x.shape == () else x, init_metrics)
init_metrics["iteration"] = jnp.array([0], dtype=jnp.int32)
init_metrics["time"] = jnp.array([0.0])  # No time recorded for initialization

# Convert init_metrics to match the metrics dictionary structure
metrics = jax.tree.map(lambda metric, init_metric: jnp.concatenate([metric, init_metric], axis=0), metrics, init_metrics)
# Jit the update function for faster iterations
update_fn = jax.jit(map_elites.update)

# Run MAP-Elites loop
for i in range(num_iterations):
    start_time = time.time()
    key, subkey = jax.random.split(key)
    (repertoire, emitter_state, current_metrics,) = update_fn(
        repertoire,
        emitter_state,
        subkey,
    )
    timelapse = time.time() - start_time

    current_metrics["iteration"] = jnp.array([1 + i], dtype=jnp.int32)
    current_metrics["time"] = jnp.array([timelapse], dtype=jnp.float32)
    
    # Convert scalar metrics to 1D arrays for concatenation
    current_metrics = jax.tree.map(lambda x: jnp.array([x]) if x.shape == () else x, current_metrics)
    metrics = jax.tree.map(lambda metric, current_metric: jnp.concatenate([metric, current_metric], axis=0), metrics, current_metrics)

# Get contents of repertoire
# repertoire.genotypes, repertoire.fitnesses, repertoire.descriptors

env_steps = metrics["iteration"]

fig, axes = plot_map_elites_results(env_steps=env_steps, metrics=metrics, repertoire=repertoire, min_descriptor=min_descriptor, max_descriptor=max_descriptor)
plt.show()

best_idx = jnp.argmax(repertoire.fitnesses)
best_fitness = jnp.max(repertoire.fitnesses)
best_descriptor = repertoire.descriptors[best_idx]

print(
    f"Best fitness in the repertoire: {best_fitness:.2f}\n",
    f"Descriptor of the best individual in the repertoire: {best_descriptor}\n",
    f"Index in the repertoire of this individual: {best_idx}\n"
)