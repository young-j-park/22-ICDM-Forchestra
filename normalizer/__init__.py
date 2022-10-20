
from normalizer.base import BaseTSNormalizer
from normalizer.standard import StandardTSNormalizer


def build_normalizer(
        normalization_method: str,
        output_rectifier: str,
) -> BaseTSNormalizer:
    if normalization_method == 'local_standard':
        normalizer = StandardTSNormalizer(
            local_updater=True,
            clip_input=None,
            output_rectifier=output_rectifier,
        )
    else:
        raise NotImplementedError

    initialize_noramlizer(normalizer)
    return normalizer


def initialize_noramlizer(normalizer):
    if not normalizer.local_updater:
        raise NotImplementedError
