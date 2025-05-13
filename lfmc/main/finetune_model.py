import argparse
import json
import logging
import tempfile
from pathlib import Path

from galileo.data.config import NORMALIZATION_DICT_FILENAME
from galileo.data.dataset import Dataset, Normalizer
from galileo.utils import device
from lfmc.core.copy import copy_dir
from lfmc.core.encoder_loader import load_from_folder
from lfmc.core.eval import finetune_and_evaluate
from lfmc.core.splits import DEFAULT_TEST_FOLDS, DEFAULT_VALIDATION_FOLDS

logger = logging.getLogger(__name__)


def load_normalizer(config_dir: Path) -> Normalizer:
    normalization_dicts = Dataset.load_normalization_values(config_dir / NORMALIZATION_DICT_FILENAME)
    return Normalizer(std=True, normalizing_dicts=normalization_dicts)


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser("Fine tune and evaluate the LFMC model")
    parser.add_argument(
        "--pretrained-models-folder",
        type=Path,
        default=Path("data/models"),
        required=True,
    )
    parser.add_argument(
        "--pretrained-model-name",
        choices=set(["base", "nano", "tiny"]),
        required=True,
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        default=Path("data/models"),
        required=True,
    )
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
        "--h5pys-only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--output-hw",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--output-timesteps",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--load-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--excluded-bands",
        type=str,
        help="The bands to exclude, separated by commas",
    )
    parser.add_argument(
        "--validation-state-regions",
        type=str,
        help="The state regions to use for validation, separated by commas",
    )
    parser.add_argument(
        "--test-state-regions",
        type=str,
        help="The state regions to use for testing, separated by commas",
    )
    args = parser.parse_args()

    logger.info("Device: %s", device)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        if args.h5pys_only:
            data_folder = args.data_folder
        else:
            data_folder = tmp_path / "data"
            data_folder.mkdir(parents=True, exist_ok=False)
            copy_dir(args.data_folder, data_folder)

        if args.h5pys_only:
            h5py_folder = tmp_path / "h5py"
            h5py_folder.mkdir(parents=True, exist_ok=False)
            copy_dir(args.h5py_folder, h5py_folder)
        else:
            # Use the original h5py folder so H5py files are saved
            h5py_folder = args.h5py_folder

        pretrained_model = load_from_folder(
            args.pretrained_models_folder / args.pretrained_model_name,
            load_weights=args.load_weights,
        )
        excluded_bands = (
            frozenset([x.strip() for x in args.excluded_bands.split(",")]) if args.excluded_bands else frozenset()
        )
        results, df = finetune_and_evaluate(
            normalizer=load_normalizer(args.config_dir),
            pretrained_model=pretrained_model,
            data_folder=data_folder,
            h5py_folder=h5py_folder,
            output_folder=args.output_folder,
            h5pys_only=args.h5pys_only,
            output_hw=args.output_hw,
            output_timesteps=args.output_timesteps,
            patch_size=args.patch_size,
            validation_folds=DEFAULT_VALIDATION_FOLDS if not args.validation_state_regions else None,
            test_folds=DEFAULT_TEST_FOLDS if not args.test_state_regions else None,
            validation_state_regions=args.validation_state_regions.split(",")
            if args.validation_state_regions
            else None,
            test_state_regions=args.test_state_regions.split(",") if args.test_state_regions else None,
            excluded_bands=excluded_bands,
        )

        logger.info("Results:\n%s", json.dumps(results, indent=4))
        with open(args.output_folder / "results.json", "w") as f:
            json.dump(results, f)
        df.to_csv(args.output_folder / "results.csv", index=False)


if __name__ == "__main__":
    main()
