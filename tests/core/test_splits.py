from typing import Sequence

from lfmc.core.splits import DEFAULT_NUM_FOLDS, DEFAULT_TEST_FOLDS, DEFAULT_VALIDATION_FOLDS


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
