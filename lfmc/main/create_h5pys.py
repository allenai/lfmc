import argparse
import logging
import multiprocessing
from pathlib import Path

from galileo.data.config import NORMALIZATION_DICT_FILENAME
from galileo.data.dataset import Dataset, Normalizer
from lfmc.core.dataset import LFMCDataset

logger = logging.getLogger(__name__)


def load_normalizer(config_dir: Path) -> Normalizer:
    normalization_dicts = Dataset.load_normalization_values(config_dir / NORMALIZATION_DICT_FILENAME)
    return Normalizer(std=True, normalizing_dicts=normalization_dicts)


def process_batch(
    config_dir: Path,
    data_folder: Path,
    h5py_folder: Path,
    batch_index: int,
    total_batches: int,
):
    lfmc_dataset = LFMCDataset(
        normalizer=load_normalizer(config_dir),
        data_folder=data_folder,
        h5py_folder=h5py_folder,
        h5pys_only=False,
    )
    dataset_size = len(lfmc_dataset)
    base = dataset_size // total_batches
    remainder = dataset_size % total_batches
    start_index = batch_index * base + min(batch_index, remainder)
    end_index = start_index + base + (1 if batch_index < remainder else 0)
    for i in range(start_index, end_index):
        result = lfmc_dataset[i]
        if result is None:
            logger.error(f"None at {i}: {lfmc_dataset.tifs[i]}")


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser("Create H5pys from TIFs")
    parser.add_argument(
        "--config-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--data-folder",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--h5py-folder",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=multiprocessing.cpu_count(),
    )
    args = parser.parse_args()

    with multiprocessing.Pool(args.num_workers) as pool:
        pool.starmap(
            process_batch,
            [
                (args.config_dir, args.data_folder, args.h5py_folder, i, args.num_workers)
                for i in range(args.num_workers)
            ],
        )


if __name__ == "__main__":
    main()
