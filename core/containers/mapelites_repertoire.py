import jax
import jax.numpy as jnp
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire as OriginalMERepertoire
from typing import Optional, Tuple
from qdax.custom_types import Centroid, Descriptor, ExtraScores, Fitness, Genotype

class MapElitesRepertoire(OriginalMERepertoire):
    @classmethod
    def init(
        cls,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        centroids: Centroid,
        *args,
        extra_scores: Optional[ExtraScores] = None,
        keys_extra_scores: Tuple[str, ...] = (),
        **kwargs,
    ) -> "MapElitesRepertoire":
        """
        Initialize a MapElitesRepertoire with the custom class.
        
        This overrides the parent init method to ensure instances 
        of this custom class are returned instead of the original.
        """
        # Call the parent's init logic but return our custom class
        original_instance = super().init(
            genotypes, fitnesses, descriptors, centroids, 
            *args, extra_scores=extra_scores, keys_extra_scores=keys_extra_scores, **kwargs
        )
        
        # Create our custom instance with the same data
        return cls(
            genotypes=original_instance.genotypes,
            fitnesses=original_instance.fitnesses,
            descriptors=original_instance.descriptors,
            centroids=original_instance.centroids,
            extra_scores=original_instance.extra_scores,
            keys_extra_scores=original_instance.keys_extra_scores,
        )

    @jax.jit
    def empty(self) -> "MapElitesRepertoire":
        """
        Empty the grid from all existing individuals.

        Returns:
            An empty MapElitesRepertoire
        """

        new_fitnesses = jnp.full_like(self.fitnesses, -jnp.inf)
        new_descriptors = jnp.zeros_like(self.descriptors)
        new_genotypes = jax.tree_map(lambda x: jnp.zeros_like(x), self.genotypes)
        return MapElitesRepertoire(
            genotypes=new_genotypes,
            fitnesses=new_fitnesses,
            descriptors=new_descriptors,
            centroids=self.centroids,
            extra_scores=self.extra_scores,
            keys_extra_scores=self.keys_extra_scores,
        )