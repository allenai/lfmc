import hashlib
import os
import random

import pandas as pd

from lfmc.core.mode import Mode

DEFAULT_NUM_FOLDS = 100

DEFAULT_VALIDATION_FOLDS, DEFAULT_TEST_FOLDS = map(
    frozenset, (lambda lst: (lst[:15], lst[15:]))(lst := random.Random(42).sample(range(DEFAULT_NUM_FOLDS), 30))
)


def num_folds() -> int:
    # Used in tests, in production this should not be changed
    return int(os.environ.get("NUM_FOLDS", DEFAULT_NUM_FOLDS))


def assign_folds(df: pd.DataFrame, id_column: str, num_folds: int) -> pd.DataFrame:
    def create_prob(value: int) -> float:
        hash = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
        return int(hash[:8], 16) / 0xFFFFFFFF

    probs = df[id_column].apply(create_prob)
    df["fold"] = (probs * num_folds).astype(int)
    return df


def assign_splits(df: pd.DataFrame, validation_folds: frozenset[int], test_folds: frozenset[int]) -> pd.DataFrame:
    def map_split(row: pd.Series) -> Mode:
        if row["fold"] in validation_folds:
            return Mode.VALIDATION
        elif row["fold"] in test_folds:
            return Mode.TEST
        else:
            return Mode.TRAIN

    df["mode"] = df.apply(map_split, axis=1)
    return df
