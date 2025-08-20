from typing import Optional, Callable, Tuple, override

import jax
import jax.numpy as jnp

from qdax.core.map_elites import MAPElites
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.custom_types import (
    Centroid,
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    Metrics,
    RNGKey,
)

class ReevalMAPElites(MAPElites):
    "Custom MAPElites with re-evaluations"
    def __init__(
        self,
        num_evals: int,
        scoring_function: Optional[
            Callable[[Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores]]
        ],
        emitter: Emitter,
        metrics_function: Callable[[MapElitesRepertoire], Metrics],
        repertoire_init: Callable[
            [Genotype, Fitness, Descriptor, Centroid, Optional[ExtraScores]],
            MapElitesRepertoire,
        ] = MapElitesRepertoire.init,

    ) -> None:
        super().__init__(scoring_function, emitter, metrics_function, repertoire_init)
        
        self._num_evals = num_evals

    def update(
        self,
        repertoire: MapElitesRepertoire,
        emitter_state: Optional[EmitterState],
        key: RNGKey,
    ) -> Tuple[MapElitesRepertoire, Optional[EmitterState], Metrics]:
        """
        Performs one iteration of the MAP-Elites algorithm.
        1. A batch of genotypes is sampled in the repertoire and the genotypes
            are copied.
        2. The copies are mutated and crossed-over
        3. The obtained offsprings are scored and then added to the repertoire.


        Args:
            repertoire: the MAP-Elites repertoire
            emitter_state: state of the emitter
            key: a jax PRNG random key

        Returns:
            the updated MAP-Elites repertoire
            the updated (if needed) emitter state
            metrics about the updated repertoire
            a new jax PRNG key
        """
        if self._scoring_function is None:
            raise ValueError("Scoring function is not set.")

        # generate offsprings with the emitter
        key, subkey = jax.random.split(key)
        genotypes, extra_info = self.ask(repertoire, emitter_state, subkey)

        # scores the offsprings
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, self._num_evals)
        batched_scoring_fn = jax.vmap(lambda k: self._scoring_function(genotypes, k))

        (fitnesses, descriptors, extra_scores) = batched_scoring_fn(keys)

        avg_fitnesses = jnp.mean(fitnesses, axis=0)
        avg_descriptors = jnp.mean(descriptors, axis=0)
        avg_extra_scores = jax.tree.map(lambda x: jnp.mean(x, axis=0), extra_scores)

        repertoire, emitter_state, metrics = self.tell(
            genotypes=genotypes,
            fitnesses=avg_fitnesses,
            descriptors=avg_descriptors,
            repertoire=repertoire,
            emitter_state=emitter_state,
            extra_scores=avg_extra_scores,
            extra_info=extra_info,
        )
        return repertoire, emitter_state, metrics
