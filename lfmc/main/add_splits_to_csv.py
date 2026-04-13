import argparse
import logging
from pathlib import Path

import pandas as pd

from lfmc.core.const import LABELS_PATH
from lfmc.core.splits import (
    DEFAULT_TEST_FOLDS,
    DEFAULT_VALIDATION_FOLDS,
    SPLIT_TYPE_COLUMN,
    SplitType,
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
    columns_to_drop = [f"{split_type}_split" for split_type in SplitType]
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    if existing_columns_to_drop:
        logging.info("Dropping existing split columns: %s", existing_columns_to_drop)
        df = df.drop(columns=existing_columns_to_drop)

    # Validate required columns exist
    required_columns = list({SPLIT_TYPE_COLUMN[st] for st in SplitType})
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    for split_type in SplitType:
        column_name = f"{split_type}_split"
        id_column = SPLIT_TYPE_COLUMN[split_type]
        logging.info("Applying deterministic fold assignment for %s splits (column: %s)", split_type, id_column)
        df = assign_random_folds(df, id_column, num_folds=num_folds())
        logging.info("Assigning %s splits from folds", split_type)
        df = assign_splits_from_folds(df, DEFAULT_VALIDATION_FOLDS, DEFAULT_TEST_FOLDS)
        df = df.rename(columns={"mode": column_name})
        df = df.drop(columns=["fold"])

    split_columns = [f"{split_type}_split" for split_type in SplitType]
    original_columns = [col for col in df.columns if col not in split_columns]
    df = df[original_columns + split_columns]

    logging.info("Writing CSV file with splits: %s", output_csv_path)
    df.to_csv(output_csv_path, index=False)

    for split_type in SplitType:
        column_name = f"{split_type}_split"
        logging.info("%s split distribution:", split_type.value.capitalize())
        split_counts = df[column_name].value_counts()
        for split, count in split_counts.items():
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
