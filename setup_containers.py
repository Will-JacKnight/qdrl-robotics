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

CONTAINER_REEVAL_ARCHIVE = [
    "Archive-Sampling",
]

CONTAINER_EXTRACT_PROPORTION_RESAMPLE = [
    "Extract-MAP-Elites",
]

def _get_add_evals_per_iter(args: Any) -> int:
    """
    Return the number of additional evaluations independent
    from emitter or extraction caused by the container at hand.
    """
    add_evals_per_iter = 0
    if args.container in CONTAINER_REEVAL_ARCHIVE:
        add_evals_per_iter += (
            args.num_centroids * args.depth * args.as_repertoire_num_samples
        )
    return add_evals_per_iter

def get_evals_per_offspring(args: Any) -> int:
    """Get the number of samples spent on each offspring."""

    # Base is the number of samples
    evals_per_offspring = args.num_samples

    # DCRL case
    if args.algo_type == "dcrl":
        # Those appraoch cannot have a sample-batch-size greater than 256
        evals_per_offspring = max(args.num_samples, args.sampling_size // 256)

    return evals_per_offspring


def get_batch_size(
    args: Any, sampling_size: int, evals_per_offspring: int
) -> Tuple[int, int, int, int]:
    """
    From the sampling_size and evals_per_offspring, return all the
    other information about sampling.

    Args:
        sampling_size
        evals_per_offspring
    Returns:
        sample_batch_size: the overall sample batch size
        init_sample_batch_size: the sample batch size of the first iteration
        emit_batch_size: the effective batch size of the
            emitter for the following iterations
        real_evals_per_iter: the number of evaluations per iteration
    """

    # Compute evals per extract
    evals_per_extract = args.num_samples
    if args.container in CONTAINER_EXTRACT_PROPORTION_RESAMPLE:
        evals_per_extract = args.as_repertoire_num_samples

    # Compute additional evals per iteration
    add_evals_per_iter = _get_add_evals_per_iter(args=args)
    assert sampling_size > add_evals_per_iter, (
        "!!!ERROR!!! Missing sampling credit for evaluation, not enough for"
        + str(add_evals_per_iter)
        + "additional evaluations."
    )
    left_sampling_size = sampling_size - add_evals_per_iter

    # Infer init_sample_batch_size
    init_sample_batch_size = sampling_size // evals_per_offspring

    # Infer sample_batch_size, real_evals_per_iter and emit_batch_size
    if args.container in CONTAINER_EXTRACT_PROPORTION_RESAMPLE:
        extract_sampling_size = int(
            left_sampling_size * args.extract_proportion_resample
        )
        extract_sampling_size = min(extract_sampling_size, args.extract_cap_resample)
        assert extract_sampling_size >= evals_per_extract, (
            "!!!ERROR!!! Missing sampling credit for evaluation, not enough left for"
            + evals_per_extract
            + "eval per extract."
        )

        effective_sampling_size = left_sampling_size - extract_sampling_size
        assert effective_sampling_size >= evals_per_offspring, (
            "!!!ERROR!!! Missing sampling credit for evaluation, not enough left for"
            + str(evals_per_offspring)
            + "eval per offspring."
        )

        emit_batch_size = effective_sampling_size // evals_per_offspring
        extract_batch_size = extract_sampling_size // evals_per_extract

        sample_batch_size = emit_batch_size + extract_batch_size
        real_evals_per_iter = (
            emit_batch_size * evals_per_offspring
            + extract_batch_size * evals_per_extract
            + add_evals_per_iter
        )
        print("\nFor Extract:")
        print(
            f"Emitting {emit_batch_size} offspring sampled {evals_per_offspring} times."
        )
        print(f"Extracting {extract_batch_size} sampled {evals_per_extract} times.")

    else:
        assert left_sampling_size >= evals_per_offspring, (
            "!!!ERROR!!! Missing sampling credit for evaluation, not enough left for"
            + str(evals_per_offspring)
            + "eval per offspring."
        )
        sample_batch_size = int(left_sampling_size // evals_per_offspring)
        emit_batch_size = sample_batch_size
        real_evals_per_iter = sample_batch_size * evals_per_offspring + add_evals_per_iter

    return sample_batch_size, init_sample_batch_size, emit_batch_size, real_evals_per_iter

def get_sampling_size(
    args: Any, batch_size: int, evals_per_offspring: int
) -> Tuple[int, int, int, int]:
    """
    From the batch_size and evals_per_offspring, return all the
    other information about sampling.

    Args:
        sampling_size
        evals_per_offspring
    Returns:
        sampling_size: the overall sampling-size of the algorithm
        init_batch_size: the batch size of the first iteration
        effective_batch_size: the effective batch size of the
            emitter for the following iterations
        real_evals_per_iter: the number of evaluations per iteration
    """

    # Compute evals per extract
    evals_per_extract = args.num_samples
    if args.container in CONTAINER_EXTRACT_PROPORTION_RESAMPLE:
        evals_per_extract = args.as_repertoire_num_samples

    add_evals_per_iter = _get_add_evals_per_iter(args=args)

    # Infer the effective batch-size
    effective_batch_size = batch_size
    extract_batch_size = int(batch_size * args.extract_proportion_resample)
    extract_batch_size = min(extract_batch_size, args.extract_cap_resample)
    effective_batch_size = batch_size - extract_batch_size

    # Infer the sampling-size and real_evals_per_iter
    if args.container in CONTAINER_EXTRACT_PROPORTION_RESAMPLE:
        sampling_size = (
            effective_batch_size * evals_per_offspring
            + (batch_size - effective_batch_size) * evals_per_extract
            + add_evals_per_iter
        )
    else:
        sampling_size = batch_size * evals_per_offspring + add_evals_per_iter
    real_evals_per_iter = sampling_size

    # Infer init_batch_size
    init_batch_size = batch_size

    return sampling_size, init_batch_size, effective_batch_size, real_evals_per_iter

def setup_container(
    container: str,
    emitter: Emitter,
    num_samples: int,
    depth: int,
    scoring_function: Callable,
    metrics_function: Callable,
    sample_batch_size: int,
    emit_batch_size: int,
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

    match container:
        case "mapelites_sampling":
            map_elites = ReevalMAPElites(
                num_samples=num_samples,
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
        case "extract_mapelites":
            map_elites = ExtractMAPElites(
                scoring_function=scoring_function,
                emitter=emitter,
                metrics_function=metrics_function,
                depth=depth,
                batch_size=sample_batch_size,
                emit_batch_size=emit_batch_size,
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
                extract_type=extract_type,
            )
    
    return map_elites, key