from typing import Sequence

from lfmc.core.const import Column
from lfmc.core.splits import (
    DEFAULT_NUM_FOLDS,
    DEFAULT_TEST_FOLDS,
    DEFAULT_VALIDATION_FOLDS,
    SPLIT_TYPE_COLUMN,
    SplitType,
)


def assert_sets_unique(sets: Sequence[set[float]]):
    assert len(sets) == len({frozenset(s) for s in sets})


def test_default_validation_and_test_folds():
    assert DEFAULT_VALIDATION_FOLDS.isdisjoint(DEFAULT_TEST_FOLDS)
    assert len(DEFAULT_VALIDATION_FOLDS) == 15
    assert len(DEFAULT_TEST_FOLDS) == 15
    for fold in DEFAULT_VALIDATION_FOLDS:
        assert 0 <= fold < DEFAULT_NUM_FOLDS
    for fold in DEFAULT_TEST_FOLDS:
        assert 0 <= fold < DEFAULT_NUM_FOLDS


def test_split_type_column_mapping():
    assert SPLIT_TYPE_COLUMN[SplitType.RANDOM] == Column.SORTING_ID
    assert SPLIT_TYPE_COLUMN[SplitType.SPATIAL] == Column.SITE_NAME
    assert set(SPLIT_TYPE_COLUMN.keys()) == set(SplitType)
