from dataclasses import dataclass
from pathlib import Path

import numpy as np
from frozendict import frozendict
from frozenlist import FrozenList
from tqdm import tqdm
from typing_extensions import override

from galileo.data.dataset import (
    SPACE_BAND_GROUPS_IDX,
    SPACE_TIME_BANDS_GROUPS_IDX,
    STATIC_BAND_GROUPS_IDX,
    TIME_BAND_GROUPS_IDX,
    Dataset,
    Normalizer,
)
from galileo.masking import MaskedOutput
from galileo.utils import masked_output_np_to_tensor
from lfmc.core.bands import SPACE_BANDS, SPACE_TIME_BANDS, STATIC_BANDS, TIME_BANDS
from lfmc.core.const import LABELS_PATH, MAX_LFMC_VALUE, Column, FileSuffix
from lfmc.core.filter import Filter, apply_filter
from lfmc.core.labels import read_labels
from lfmc.core.mode import Mode
from lfmc.core.padding import pad_dates
from lfmc.core.splits import assign_folds, assign_splits, num_folds


@dataclass(frozen=True)
class SampleData:
    sorting_id: int
    start_month: int
    latitude: float
    longitude: float
    lfmc_value: float


class LFMCDataset(Dataset):
    def __init__(
        self,
        normalizer: Normalizer,
        data_folder: Path,
        h5py_folder: Path,
        download: bool = False,
        h5pys_only: bool = False,
        output_hw: int = 32,
        output_timesteps: int = 12,
        space_time_bands: FrozenList[str] = FrozenList(SPACE_TIME_BANDS),
        space_bands: FrozenList[str] = FrozenList(SPACE_BANDS),
        time_bands: FrozenList[str] = FrozenList(TIME_BANDS),
        static_bands: FrozenList[str] = FrozenList(STATIC_BANDS),
        mode: Mode | None = None,
        validation_folds: frozenset[int] | None = None,
        test_folds: frozenset[int] | None = None,
        filter: Filter | None = None,
    ):
        super().__init__(
            data_folder,
            download,
            h5py_folder,
            h5pys_only,
            output_hw,
            output_timesteps,
        )
        self.normalizer = normalizer
        self.masks = self.make_masks(
            output_hw,
            output_timesteps,
            space_time_bands,
            space_bands,
            time_bands,
            static_bands,
        )

        self.mode = mode
        self.validation_folds = validation_folds
        self.test_folds = test_folds

        self.tifs: list[Path] = []
        self.h5pys: list[Path] = []
        stem_to_sample: dict[str, SampleData] = {}

        data = apply_filter(read_labels(LABELS_PATH), filter)
        if mode is not None:
            if validation_folds is None:
                raise ValueError("validation_folds must be provided if mode is provided")
            if test_folds is None:
                raise ValueError("test_folds must be provided if mode is provided")
            data = assign_folds(data, Column.SORTING_ID, num_folds=num_folds())
            data = assign_splits(data, validation_folds, test_folds)
            data = data[data["mode"] == mode]

        suffix = FileSuffix.H5 if h5pys_only else FileSuffix.TIF
        folder = h5py_folder if h5pys_only else data_folder
        for _, row in tqdm(data.iterrows(), total=len(data)):
            filepath = folder / f"{row[Column.SORTING_ID]}.{suffix}"
            if filepath.exists():
                if filepath.suffix == f".{FileSuffix.H5}":
                    self.h5pys.append(filepath)
                else:
                    self.tifs.append(filepath)
                start_date, _ = pad_dates(row[Column.SAMPLING_DATE].date())
                stem_to_sample[filepath.stem] = SampleData(
                    sorting_id=row[Column.SORTING_ID],
                    latitude=row[Column.LATITUDE],
                    longitude=row[Column.LONGITUDE],
                    lfmc_value=row[Column.LFMC_VALUE],
                    start_month=start_date.month,
                )

        self.stem_to_sample: frozendict[str, SampleData] = frozendict(stem_to_sample)

    @override
    def month_array_from_file(self, tif_path: Path, num_timesteps: int) -> np.ndarray:
        sample_data = self.stem_to_sample[tif_path.stem]
        start_month = sample_data.start_month
        return np.fmod(np.arange(start_month - 1, start_month - 1 + num_timesteps), 12)

    @override
    @staticmethod
    def subset_image(
        space_time_x: np.ndarray,
        space_x: np.ndarray,
        time_x: np.ndarray,
        static_x: np.ndarray,
        months: np.ndarray,
        size: int,
        num_timesteps: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not ((space_time_x.shape[0] == space_x.shape[0]) & (space_time_x.shape[1] == space_x.shape[1])):
            raise ValueError("Disagreement in height and width of space_time_x and space_x")
        if space_time_x.shape[2] != time_x.shape[0]:
            raise ValueError("Disagreement in number of timesteps of space_time_x and time_x")
        if not ((space_time_x.shape[0] >= size) & (space_time_x.shape[1] >= size)):
            raise ValueError(f"size {size} > height {space_time_x.shape[0]} or width {space_time_x.shape[1]}")
        if space_time_x.shape[2] < num_timesteps:
            raise ValueError(f"num_timesteps {num_timesteps} > time {space_time_x.shape[2]}")
        start_h = int((space_time_x.shape[0] - size) / 2)
        start_w = int((space_time_x.shape[1] - size) / 2)
        if not ((start_h >= 0) & (start_w >= 0)):
            raise ValueError(f"start_h {start_h} or start_w {start_w} < 0")
        total_timesteps = space_time_x.shape[2]
        timesteps_to_sample = range(total_timesteps - num_timesteps, total_timesteps)

        return (
            space_time_x[start_h : start_h + size, start_w : start_w + size, timesteps_to_sample],
            space_x[start_h : start_h + size, start_w : start_w + size],
            time_x[timesteps_to_sample],
            static_x,
            months[timesteps_to_sample],
        )

    def __len__(self) -> int:
        return len(self.tifs) if not self.h5pys_only else len(self.h5pys)

    def __getitem__(self, idx: int) -> tuple[MaskedOutput, tuple[float, float], float]:
        if self.h5pys_only:
            (s_t_x, sp_x, t_x, st_x, months) = self.read_and_slice_h5py_file(self.h5pys[idx]).normalize(self.normalizer)
            filepath = self.h5pys[idx]
        else:
            (s_t_x, sp_x, t_x, st_x, months) = self.load_tif(idx).normalize(self.normalizer)
            filepath = self.tifs[idx]

        sample_data = self.stem_to_sample[filepath.stem]
        normalized_lfmc_value = min(sample_data.lfmc_value, MAX_LFMC_VALUE) / MAX_LFMC_VALUE
        return (
            masked_output_np_to_tensor(s_t_x, sp_x, t_x, st_x, *self.masks, months),
            (sample_data.latitude, sample_data.longitude),
            normalized_lfmc_value,
        )

    @override
    @staticmethod
    def return_subset_indices(
        total_h,
        total_w,
        total_t,
        size: int,
        num_timesteps: int,
    ) -> tuple[int, int, int]:
        """
        Differs from the parent function because subsetting is always centered and time cuts always remove from the end.
        """
        start_t = total_t - num_timesteps
        start_h = int((total_h - size) / 2)
        start_w = int((total_w - size) / 2)
        return start_h, start_w, start_t

    def make_masks(
        self,
        output_hw: int,
        output_timesteps: int,
        space_time_bands: FrozenList[str],
        space_bands: FrozenList[str],
        time_bands: FrozenList[str],
        static_bands: FrozenList[str],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        space_time_mask = np.ones(
            [
                output_hw,
                output_hw,
                output_timesteps,
                len(SPACE_TIME_BANDS_GROUPS_IDX),
            ],
        )
        space_time_include = [i for i, key in enumerate(SPACE_TIME_BANDS_GROUPS_IDX) if key in space_time_bands]
        space_time_mask[:, :, :, space_time_include] = 0

        space_mask = np.ones([output_hw, output_hw, len(SPACE_BAND_GROUPS_IDX)])
        space_include = [i for i, key in enumerate(SPACE_BAND_GROUPS_IDX) if key in space_bands]
        space_mask[:, :, space_include] = 0

        time_mask = np.ones([output_timesteps, len(TIME_BAND_GROUPS_IDX)])
        time_include = [i for i, key in enumerate(TIME_BAND_GROUPS_IDX) if key in time_bands]
        time_mask[:, time_include] = 0

        static_mask = np.ones([len(STATIC_BAND_GROUPS_IDX)])
        static_include = [i for i, key in enumerate(STATIC_BAND_GROUPS_IDX) if key in static_bands]
        static_mask[static_include] = 0

        return space_time_mask, space_mask, time_mask, static_mask
