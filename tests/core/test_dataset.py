from pathlib import Path
from typing import Any, Sequence

import pytest

from galileo.data.dataset import Normalizer
from lfmc.core.const import MeteorologicalSeason, WorldCoverClass
from lfmc.core.dataset import LFMCDataset
from lfmc.core.filter import Filter
from lfmc.core.mode import Mode
from lfmc.core.splits import num_folds


def assert_sets_unique(sets: Sequence[set[Any]]):
    assert len(sets) == len({frozenset(s) for s in sets})


def test_dataset_all_samples(data_folder: Path, h5py_folder: Path, normalizer: Normalizer):
    dataset = LFMCDataset(
        normalizer=normalizer,
        data_folder=data_folder,
        h5py_folder=h5py_folder,
        h5pys_only=False,
    )
    assert len(dataset) == 10


def test_dataset_train_validation_test_splits(data_folder: Path, h5py_folder: Path, normalizer: Normalizer):
    def create_dataset(mode: Mode, validation_folds: frozenset[int], test_folds: frozenset[int]):
        return LFMCDataset(
            normalizer=normalizer,
            data_folder=data_folder,
            h5py_folder=h5py_folder,
            h5pys_only=False,
            mode=mode,
            validation_folds=validation_folds,
            test_folds=test_folds,
        )

    validation_folds = frozenset([0])
    test_folds = frozenset([1])
    train_dataset = create_dataset(Mode.TRAIN, validation_folds, test_folds)
    validation_dataset = create_dataset(Mode.VALIDATION, validation_folds, test_folds)
    test_dataset = create_dataset(Mode.TEST, validation_folds, test_folds)

    training_samples = {train_dataset[i][1] for i in range(len(train_dataset))}
    validation_samples = {validation_dataset[i][1] for i in range(len(validation_dataset))}
    test_samples = {test_dataset[i][1] for i in range(len(test_dataset))}
    assert_sets_unique([training_samples, validation_samples, test_samples])


def test_dataset_splits_are_different(data_folder: Path, h5py_folder: Path, normalizer: Normalizer):
    def create_dataset(mode: Mode, validation_folds: frozenset[int], test_folds: frozenset[int]):
        return LFMCDataset(
            normalizer=normalizer,
            data_folder=data_folder,
            h5py_folder=h5py_folder,
            h5pys_only=False,
            mode=mode,
            validation_folds=validation_folds,
            test_folds=test_folds,
        )

    training_samples_by_split_id: dict[int, set[tuple[float, float]]] = {}
    validation_samples_by_split_id: dict[int, set[tuple[float, float]]] = {}
    test_samples_by_split_id: dict[int, set[tuple[float, float]]] = {}
    for validation_fold in range(num_folds()):
        test_fold = (validation_fold + 1) % num_folds()
        train_dataset = create_dataset(Mode.TRAIN, frozenset({validation_fold}), frozenset({test_fold}))
        validation_dataset = create_dataset(Mode.VALIDATION, frozenset({validation_fold}), frozenset({test_fold}))
        test_dataset = create_dataset(Mode.TEST, frozenset({validation_fold}), frozenset({test_fold}))

        for i in range(len(train_dataset)):
            _, (latitude, longitude), _ = train_dataset[i]
            training_samples_by_split_id.setdefault(validation_fold, set()).add((latitude, longitude))
        for i in range(len(validation_dataset)):
            _, (latitude, longitude), _ = validation_dataset[i]
            validation_samples_by_split_id.setdefault(validation_fold, set()).add((latitude, longitude))
        for i in range(len(test_dataset)):
            _, (latitude, longitude), _ = test_dataset[i]
            test_samples_by_split_id.setdefault(validation_fold, set()).add((latitude, longitude))

    assert_sets_unique(list(training_samples_by_split_id.values()))
    assert_sets_unique(list(validation_samples_by_split_id.values()))
    assert_sets_unique(list(test_samples_by_split_id.values()))


@pytest.mark.parametrize(
    "season, expected_len",
    [
        (None, 10),
        (MeteorologicalSeason.WINTER, 1),
        (MeteorologicalSeason.SPRING, 1),
        (MeteorologicalSeason.SUMMER, 5),
        (MeteorologicalSeason.AUTUMN, 3),
    ],
)
def test_dataset_filter_seasons(
    data_folder: Path,
    h5py_folder: Path,
    normalizer: Normalizer,
    season: MeteorologicalSeason,
    expected_len: int,
):
    dataset = LFMCDataset(
        normalizer=normalizer,
        data_folder=data_folder,
        h5py_folder=h5py_folder,
        h5pys_only=False,
        filter=Filter(seasons={season} if season is not None else None),
    )
    assert len(dataset) == expected_len


@pytest.mark.parametrize(
    "landcover, expected_len",
    [
        (None, 10),
        (WorldCoverClass.TREE_COVER, 5),
        (WorldCoverClass.SHRUBLAND, 1),
        (WorldCoverClass.GRASSLAND, 4),
        (WorldCoverClass.CROPLAND, 0),
        (WorldCoverClass.BUILT_UP, 0),
        (WorldCoverClass.BARE_VEGETATION, 0),
        (WorldCoverClass.SNOW_AND_ICE, 0),
        (WorldCoverClass.WATER, 0),
        (WorldCoverClass.HERBACEOUS_WETLAND, 0),
        (WorldCoverClass.MANGROVES, 0),
        (WorldCoverClass.MOSS_AND_LICHEN, 0),
    ],
)
def test_dataset_filter_landcover(
    data_folder: Path,
    h5py_folder: Path,
    normalizer: Normalizer,
    landcover: WorldCoverClass,
    expected_len: int,
):
    dataset = LFMCDataset(
        normalizer=normalizer,
        data_folder=data_folder,
        h5py_folder=h5py_folder,
        h5pys_only=False,
        filter=Filter(landcover={landcover} if landcover is not None else None),
    )
    assert len(dataset) == expected_len


@pytest.mark.parametrize(
    "elevation, expected_len",
    [
        (None, 10),
        ((0, 1000), 1),
        ((1000, 2000), 7),
        ((2000, 3000), 2),
        ((3000, 4000), 0),
    ],
)
def test_dataset_filter_elevation(
    data_folder: Path,
    h5py_folder: Path,
    normalizer: Normalizer,
    elevation: tuple[int, int],
    expected_len: int,
):
    dataset = LFMCDataset(
        normalizer=normalizer,
        data_folder=data_folder,
        h5py_folder=h5py_folder,
        h5pys_only=False,
        filter=Filter(elevation=elevation),
    )
    assert len(dataset) == expected_len


@pytest.mark.parametrize(
    "high_fire_danger, expected_len",
    [
        (None, 10),
        (True, 8),
        (False, 2),
    ],
)
def test_dataset_filter_high_fire_danger(
    data_folder: Path,
    h5py_folder: Path,
    normalizer: Normalizer,
    high_fire_danger: bool,
    expected_len: int,
):
    dataset = LFMCDataset(
        normalizer=normalizer,
        data_folder=data_folder,
        h5py_folder=h5py_folder,
        h5pys_only=False,
        filter=Filter(high_fire_danger=high_fire_danger),
    )
    assert len(dataset) == expected_len
