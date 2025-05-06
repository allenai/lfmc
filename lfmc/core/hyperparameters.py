from dataclasses import dataclass


@dataclass(frozen=True)
class HyperParameters:
    batch_size: int
    num_workers: int


DEFAULT_HYPERPARAMETERS = HyperParameters(batch_size=16, num_workers=0)
