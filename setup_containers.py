from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from jax.lax import switch
from qdax.core.containers.repertoire import Repertoire
from qdax.core.emitters.emitter import Emitter
from qdax.core.map_elites import MAPElites
from qdax.core.mels import MELS
from qdax.custom_types import Metrics, RNGKey

from core.mapelites_sampling import ReevalMAPElites
from core.archive_sampling import ArchiveSampling
from core.extract_map_elites import ExtractMAPElites

from core.sampling import average, closest, iqr, mad, median, mode, std

# Extractor list
EXTRACTOR_LIST = {
    "Average": average,
    "Median": median,
    "Mode": mode,
    "Closest": closest,
    "STD": std,
    "MAD": mad,
    "IQR": iqr,
}

def setup_container(
    container_name: str,
    emitter: Emitter,
    num_samples: int,
    depth: int,
    scoring_function: Callable,
    metrics_function: Callable,
    sampling_size: int,
    max_number_evals: int,
    as_repertoire_num_samples: int,
    fitness_extractor: str,
    fitness_reproducibility_extractor: str,
    descriptor_extractor: str,
    descriptor_reproducibility_extractor: str,
    extract_type: str,
    key: RNGKey,
) -> Tuple[MAPElites, RNGKey]:

    extract_proportion = 0.25

    match container_name:
        case "mapelites_sampling":
            map_elites = ReevalMAPElites(
                num_evals=num_samples,
                scoring_function=scoring_function,
                emitter=emitter,
                metrics_function=metrics_function,
            )
        case "archive_sampling":
            map_elites = ArchiveSampling(
                scoring_function=scoring_function,
                emitter=emitter,
                metrics_function=metrics_function,
                depth=depth,
                max_number_evals=max_number_evals,
                num_samples=num_samples,
                repertoire_num_samples=as_repertoire_num_samples,
                fitness_extractor=EXTRACTOR_LIST[fitness_extractor],
                fitness_reproducibility_extractor=EXTRACTOR_LIST[
                    fitness_reproducibility_extractor
                ],
                descriptor_extractor=EXTRACTOR_LIST[descriptor_extractor],
                descriptor_reproducibility_extractor=EXTRACTOR_LIST[
                    descriptor_reproducibility_extractor
                ],
            )
        # case "extract_mapelites":
        #     map_elites = ExtractMAPElites(
        #         scoring_function=scoring_fn,
        #         emitter=emitter,
        #         metrics_function=metrics_fn,
        #         depth=depth,
        #         batch_size=batch_size,
        #         emit_batch_size=effective_batch_size,
        #         max_number_evals=max_number_evals,
        #         num_samples=num_samples,
        #         repertoire_num_samples=as_repertoire_num_samples,
        #         fitness_extractor=EXTRACTOR_LIST[fitness_extractor],
        #         fitness_reproducibility_extractor=EXTRACTOR_LIST[
        #             fitness_reproducibility_extractor
        #         ],
        #         descriptor_extractor=EXTRACTOR_LIST[descriptor_extractor],
        #         descriptor_reproducibility_extractor=EXTRACTOR_LIST[
        #             descriptor_reproducibility_extractor
        #         ],
        #         extract_type=extract_type,
        #     )
    
    return map_elites, key