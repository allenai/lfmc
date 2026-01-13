import argparse
import logging
from pathlib import Path

import pandas as pd

from lfmc.core.const import LABELS_PATH, Column
from lfmc.core.splits import (
    DEFAULT_TEST_FOLDS,
    DEFAULT_VALIDATION_FOLDS,
    assign_random_folds,
    assign_splits_from_folds,
    num_folds,
)


def add_splits_to_csv(input_csv_path: Path, output_csv_path: Path) -> None:
    """Add train/validation/test split assignments to the CSV file.

    Uses the same deterministic split scheme as training:
    - random_split: Assigns folds based on sorting_id
    - spatial_split: Assigns folds based on site_name
    """
    logging.info("Reading CSV file: %s", input_csv_path)
    df = pd.read_csv(input_csv_path)

    # Drop existing split columns if they exist
    columns_to_drop = ["random_split", "spatial_split"]
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    if existing_columns_to_drop:
        logging.info("Dropping existing split columns: %s", existing_columns_to_drop)
        df = df.drop(columns=existing_columns_to_drop)

    # Validate required columns exist
    required_columns = [Column.SORTING_ID, Column.SITE_NAME]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Compute random_split
    logging.info("Applying deterministic fold assignment for random splits")
    df = assign_random_folds(df, Column.SORTING_ID, num_folds=num_folds())
    logging.info("Assigning random splits from folds")
    df = assign_splits_from_folds(df, DEFAULT_VALIDATION_FOLDS, DEFAULT_TEST_FOLDS)
    df = df.rename(columns={"mode": "random_split"})
    df = df.drop(columns=["fold"])

    # Compute spatial_split
    logging.info("Applying deterministic fold assignment for spatial splits")
    df = assign_random_folds(df, Column.SITE_NAME, num_folds=num_folds())
    logging.info("Assigning spatial splits from folds")
    df = assign_splits_from_folds(df, DEFAULT_VALIDATION_FOLDS, DEFAULT_TEST_FOLDS)
    df = df.rename(columns={"mode": "spatial_split"})
    df = df.drop(columns=["fold"])

    # Reorder columns to put random_split and spatial_split at the end
    original_columns = [col for col in df.columns if col not in ["random_split", "spatial_split"]]
    df = df[original_columns + ["random_split", "spatial_split"]]

    logging.info("Writing CSV file with splits: %s", output_csv_path)
    df.to_csv(output_csv_path, index=False)

    # Log split distribution
    logging.info("Random split distribution:")
    random_split_counts = df["random_split"].value_counts()
    for split, count in random_split_counts.items():
        logging.info("  %s: %d rows (%.2f%%)", split, count, 100 * count / len(df))

    logging.info("Spatial split distribution:")
    spatial_split_counts = df["spatial_split"].value_counts()
    for split, count in spatial_split_counts.items():
        logging.info("  %s: %d rows (%.2f%%)", split, count, 100 * count / len(df))


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser("Add train/validation/test splits to LFMC CSV file")
    parser.add_argument(
        "--input-csv-path",
        type=Path,
        default=LABELS_PATH,
        help="Path to input CSV file (default: %(default)s)",
    )
    parser.add_argument(
        "--output-csv-path",
        type=Path,
        default=LABELS_PATH,
        help="Path to output CSV file with splits added (default: overwrites input)",
    )
    args = parser.parse_args()

    add_splits_to_csv(args.input_csv_path, args.output_csv_path)


if __name__ == "__main__":
    main()
